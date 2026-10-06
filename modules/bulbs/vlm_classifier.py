import cv2
import numpy as np
import open_clip
from PIL import Image
import torch


def _pad_square(c):
    """Pads a crop to a square with its median colour so CLIP's resize +
    center-crop doesn't cut off the ends of long, thin fixtures."""
    h, w = c.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    fill = tuple(int(v) for v in np.median(c.reshape(-1, c.shape[2]), axis=0))
    return cv2.copyMakeBorder(c, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=fill)


class VLMClassifier:
    def __init__(self, model_id: str = "hf-hub:apple/MobileCLIP-B-OpenCLIP",
                 margin: float = 0.05, tight_weight: float = 0.5):
        """
        margin       : required gap between positive mass and the strongest negative.
        tight_weight : weight of the tight crop's probabilities vs. the context crop.
        """
        self.margin = margin
        self.tight_weight = tight_weight
        self.last_details = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "System CPU"

        print("\n" + "=" * 60)
        print("[VLM INIT] Initializing zero-shot verifier...")
        print(f"  • Model Checkpoint : {model_id}")
        print(f"  • Compute Target   : {self.device.upper()} ({device_name})")
        print("=" * 60)

        print("[1/2] Loading MobileCLIP model and transforms...", end=" ", flush=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id)
        self.tokenizer = open_clip.get_tokenizer(model_id)

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = self.model.to(device=self.device, dtype=dtype)
        self.model.eval()
        print("Done.")

        print("[2/2] Pre-computing text embeddings for target classes...", end=" ", flush=True)
        # (short_name, prompt)
        classes = [
            ("outdoor_lamp",
             "a close-up photo of an active glowing light fixture, outdoor garden pathway light, "
             "hanging patio lamp, landscape spotlight, or illuminated luminaire"),
            ("indoor_ceiling",
             "a photo of a switched-on ceiling light: recessed downlight, LED linear strip light, "
             "tube light, panel light, or pendant lamp glowing bright white"),
            # NEW positive: flare/starburst around a real lamp used to live in the negatives
            ("warm_lamp_glare",
             "a photo of a small bright warm-white or orange lamp or LED bulb with a starburst "
             "glare or soft glowing halo around it"),

            ("glossy_reflection",
             "a photo of specular reflection on glossy plant leaves, wet brick pathway glare, "
             "shiny chrome, polished black granite countertop, a plastic bottle highlight, "
             "or a wet floor reflection"),
            ("skin",
             "a photo of human skin, a hand, fingers, fingertip, fingernail, a person's face, "
             "forehead highlight, oily skin reflection, or facial hair"),
            ("window_daylight",
             "a photo of bright outdoor daylight, a sunlit window, glass louvers, horizontal "
             "window blinds, window slats, or a bright window frame"),
            ("inert_surface",
             "a photo of a blank white ceiling, plain drywall, dark night sky, tree branches, "
             "unlit foliage, door frame, or electrical switchboard"),
            ("foliage_highlight",
             "a photo of dark green leaves and bushes with small bright highlights, "
             "garden foliage at night"),
            ("lit_wall",
             "a photo of a brightly lit wall, a doorway showing an indoor room, "
             "or a white painted surface"),
        ]
        self.short_names = [n for n, _ in classes]
        self.candidate_labels = [p for _, p in classes]
        self.pos_idx = [0, 1, 2]
        self.neg_idx = [i for i in range(len(classes)) if i not in self.pos_idx]

        text_tokens = self.tokenizer(self.candidate_labels).to(self.device)
        with torch.no_grad():
            text_feats = self.model.encode_text(text_tokens)
            self.text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)
        print("Done.")
        print("[STATUS] VLM Engine is active and ready for inference.\n")

    def _probs(self, crops_bgr):
        pil = [Image.fromarray(_pad_square(c)[:, :, ::-1]) for c in crops_bgr]
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        x = torch.stack([self.preprocess(im) for im in pil]).to(device=self.device, dtype=dtype)
        with torch.no_grad():
            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = (f @ self.text_features.T) * self.model.logit_scale.exp()
            return logits.float().softmax(dim=-1).cpu().numpy()

    def verify_crops_batch(self, crops_bgr, scene_t: float = 0.0,
                           crops_tight=None, skin_fracs=None, halos=None):
        """
        crops_bgr   : context crops (padded box).
        crops_tight : optional tight crops of the same candidates; probabilities are
                      blended with the context crops so a downlight isn't judged on
                      "mostly ceiling" alone.
        skin_fracs  : optional per-candidate skin fraction of the surrounding ring;
                      >0.6 raises the required confidence (soft, because warm brick
                      and wood are also skin-coloured).
        halos       : optional per-candidate halo gradient; a clear bloom gives a
                      small confidence bonus, a hard-edged speck a small penalty.

        Returns [(is_bulb, conf, label)]; self.last_details holds the diagnostics.
        """
        self.last_details = []
        if not crops_bgr:
            return []

        n = len(crops_bgr)
        print(f"[VLM INFERENCE] Batch-evaluating {n} candidate crop(s) on {self.device.upper()}...",
              end=" ", flush=True)

        if crops_tight is not None:
            both = self._probs(list(crops_bgr) + list(crops_tight))
            w = self.tight_weight
            probs = (1 - w) * both[:n] + w * both[n:]
        else:
            probs = self._probs(crops_bgr)
        print("Done.")

        results = []
        for i, p in enumerate(probs):
            pos = float(sum(p[j] for j in self.pos_idx))
            neg_j = max(self.neg_idx, key=lambda j: p[j])
            best_neg = float(p[neg_j])

            h, w_ = crops_bgr[i].shape[:2]
            is_small = (h * w_) < 2500
            required = (0.45 if is_small else 0.55) - 0.08 * scene_t

            if skin_fracs is not None and skin_fracs[i] > 0.6:
                required += 0.15
            conf = pos
            if halos is not None:
                conf += 0.05 if halos[i] >= 4 else (-0.05 if halos[i] < 1 else 0.0)

            is_bulb = (conf >= required) and (pos - best_neg >= self.margin)
            self.last_details.append({
                "pos": pos, "conf": conf, "required": required,
                "best_neg": best_neg, "best_neg_name": self.short_names[neg_j],
            })
            results.append((is_bulb, conf, "bulb" if is_bulb else "artifact"))
        return results