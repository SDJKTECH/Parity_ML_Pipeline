import cv2
import numpy as np
from Pipeline.modules.bulbs.state_analyzer import detect_bulb_candidates, get_luminosity, scene_brightness
from Pipeline.modules.bulbs.vlm_classifier import VLMClassifier


def _grow(box, w, h, min_side):
    """Grow a box symmetrically around its centre to at least min_side."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    hw = max((x2 - x1) // 2, min_side // 2)
    hh = max((y2 - y1) // 2, min_side // 2)
    return max(0, cx - hw), max(0, cy - hh), min(w, cx + hw), min(h, cy + hh)


class BulbDetector:
    def __init__(self, model_id: str = "hf-hub:apple/MobileCLIP-B-OpenCLIP"):
        self.vlm = VLMClassifier(model_id=model_id)

    def detect_bulbs(self, frame, threshold_value=None, min_area=None,
                     pad_pixels: int = 45, debug: bool = False) -> dict:
        """
        Candidate extraction + zero-shot CLIP verification.
        With debug=True the result also contains `debug_frame`: accepted boxes in
        green, rejected candidates in red with the stage and reason that dropped them.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame passed to BulbDetector is None or empty.")

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w = frame.shape[:2]
        safe_pad = 45 if pad_pixels is None else int(pad_pixels)

        t = scene_brightness(get_luminosity(frame))
        safe_pad = int(round(safe_pad * (1 - 0.55 * t)))

        rejected = [] if debug else None
        candidates = detect_bulb_candidates(
            frame, threshold_value=threshold_value, min_area=min_area,
            pad_pixels=safe_pad, debug=debug, rejected=rejected,
        )

        annotated = frame.copy()
        dbg = frame.copy() if debug else None

        def finish(dets):
            if debug:
                for r in rejected:
                    x1, y1, x2, y2 = r["draw_bbox"]
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(dbg, r["reason"], (x1, max(10, y1 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
            out = {"detected": len(dets) > 0, "count": len(dets),
                   "detections": dets, "annotated_frame": annotated}
            if debug:
                out["debug_frame"] = dbg
            return out

        if not candidates:
            return finish([])

        # Two crops per candidate: context (padded) and tight (the blob itself)
        MIN_CTX, MIN_TIGHT = 80, 32
        valid, ctx_crops, tight_crops = [], [], []
        for cand in candidates:
            cx1, cy1, cx2, cy2 = _grow(cand["bbox"], w, h, MIN_CTX)
            tx1, ty1, tx2, ty2 = _grow(cand.get("draw_bbox", cand["bbox"]), w, h, MIN_TIGHT)
            ctx = frame[cy1:cy2, cx1:cx2]
            tight = frame[ty1:ty2, tx1:tx2]
            if min(ctx.shape[:2]) > 2 and min(tight.shape[:2]) > 2:
                valid.append(cand)
                ctx_crops.append(ctx)
                tight_crops.append(tight)

        if not ctx_crops:
            return finish([])

        results = self.vlm.verify_crops_batch(
            ctx_crops, scene_t=t, crops_tight=tight_crops,
            skin_fracs=[c["skin_frac"] for c in valid],
            halos=[c["halo"] for c in valid],
        )

        confirmed = []
        for cand, (is_bulb, conf, label), det in zip(valid, results, self.vlm.last_details):
            print(f"t={t:.2f} conf={conf:.2f} (need {det['required']:.2f}) "
                  f"neg={det['best_neg_name']}:{det['best_neg']:.2f} "
                  f"core={cand['core_frac']:.2f}/{cand['core_px']}px "
                  f"contrast={cand['contrast']:.0f} halo={cand['halo']:.1f} "
                  f"skin={cand['skin_frac']:.2f} mask={cand['mask']}")
            dx1, dy1, dx2, dy2 = cand["draw_bbox"]

            if not is_bulb:
                if debug:
                    rejected.append({
                        "draw_bbox": cand["draw_bbox"],
                        "reason": f"vlm {conf:.2f}/{det['required']:.2f} {det['best_neg_name']}",
                    })
                continue

            confirmed.append({
                "id": len(confirmed) + 1,
                "confidence": round(float(conf), 4),
                "label": str(label),
                "bbox": [dx1, dy1, dx2, dy2],
                "contour_area": cand.get("area", 0.0),
            })

            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            text = f"Bulb {min(conf, 1.0) * 100:.0f}%"
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            (tw, tth), base = cv2.getTextSize(text, font, fs, th)
            ty = dy1 - 6 if dy1 - 6 > tth + 4 else dy1 + tth + 6
            cv2.rectangle(annotated, (dx1, ty - tth - 2), (dx1 + tw + 2, ty + base - 1), (0, 180, 0), -1)
            cv2.putText(annotated, text, (dx1 + 1, ty), font, fs, (0, 0, 0), th, cv2.LINE_AA)
            if debug:
                cv2.rectangle(dbg, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)

        return finish(confirmed)