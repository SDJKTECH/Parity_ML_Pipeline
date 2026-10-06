import cv2
import numpy as np


def get_luminosity(image):
    """Safely converts 1-channel, 3-channel (BGR), or 4-channel (BGRA) images to grayscale."""
    if image is None:
        raise ValueError("Image passed to get_luminosity is None.")

    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        return image

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unexpected image shape: {image.shape}")


def scene_brightness(gray):
    """
    Continuous scene-brightness factor.
    0.0 = night scene, 1.0 = bright indoor / daylight scene.
    Uses the median (not the mean) so flares and big lit walls don't fool it.
    """
    return float(np.clip((float(np.median(gray)) - 50) / 70.0, 0.0, 1.0))


def reduce_haziness(gray_image, gamma=2.2):
    """Reduces haziness/glare using Gamma Correction via a Look-Up Table (LUT)."""
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray_image, table)


def isolate_active_bulbs(smoothed_image, threshold_value=225):
    """Applies binary thresholding to isolate bright glowing regions."""
    _, thresh = cv2.threshold(smoothed_image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh


def is_round_or_oval(contour, min_circularity=0.05, min_axis_ratio=0.15, min_solidity=0.55):
    """Checks if a light emission blob resembles a bulb source or lens bloom."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0 or area == 0:
        return False

    circularity = (4 * np.pi * area) / (perimeter ** 2)

    if len(contour) >= 5:
        (_, _), (d1, d2), _ = cv2.fitEllipse(contour)
        axis_ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
    else:
        _, _, w, h = cv2.boundingRect(contour)
        axis_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    return (circularity >= min_circularity) or (axis_ratio >= min_axis_ratio) or (solidity >= min_solidity)


def is_linear_light_source(contour, min_aspect_ratio=2.5, max_aspect_ratio=25.0, min_solidity=0.70):
    """Identifies straight, elongated linear light fixtures (tube lights, batten LEDs)."""
    area = cv2.contourArea(contour)
    if area < 60:
        return False

    _, _, bw, bh = cv2.boundingRect(contour)
    major = max(bw, bh)
    minor = min(bw, bh) + 1e-5
    aspect_ratio = major / minor

    if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    return solidity >= min_solidity


# YCrCb skin range (classic Chai & Ngan bounds)
_SKIN_LO = (0, 133, 77)
_SKIN_HI = (255, 173, 127)


def halo_gradient(gray, contour):
    """
    Emitters bloom: brightness just outside the blob (band A) is clearly higher
    than brightness a bit further out (band B). A glint on a bottle / granite /
    fingertip is a hard-edged speck, so A ~ B.
    Returns mean(A) - mean(B) in gray levels (higher = more emitter-like).
    """
    H, W = gray.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    R = int(np.clip(min(bw, bh) // 2, 3, 15))
    pad = 3 * R + 2
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + bw + pad), min(H, y + bh + pad)

    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1, offset=(-x1, -y1))

    def grow(r):
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0

    inner = cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0
    band_a = grow(R) & ~inner
    band_b = grow(3 * R) & ~grow(2 * R)
    if not band_a.any() or not band_b.any():
        return 0.0
    roi = gray[y1:y2, x1:x2]
    return float(roi[band_a].mean() - roi[band_b].mean())


def blob_light_stats(image, gray, contour, white_thr=235):
    
    H, W = gray.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)

    r = min(max(6, int(0.75 * min(bw, bh)) + 6), 30)

    x1, y1 = max(0, x - r - 2), max(0, y - r - 2)
    x2, y2 = min(W, x + bw + r + 2), min(H, y + bh + r + 2)

    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1, offset=(-x1, -y1))

    inner = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    outer = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    inside = mask > 0
    ring = (outer > 0) & (inner == 0)

    roi_gray = gray[y1:y2, x1:x2]
    skin_frac = 0.0
    if image.ndim == 3:
        roi_bgr = np.ascontiguousarray(image[y1:y2, x1:x2, :3])
        mid = np.sort(roi_bgr, axis=2)[:, :, 1]            # median of B,G,R
        if ring.any():
            skin = cv2.inRange(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb), _SKIN_LO, _SKIN_HI) > 0
            skin_frac = float(skin[ring].mean())
    else:
        mid = roi_gray

    n_in = int(inside.sum())
    core_px = int((mid[inside] >= white_thr).sum()) if n_in else 0
    core_frac = core_px / n_in if n_in else 0.0
    contrast = float(roi_gray[inside].mean() - roi_gray[ring].mean()) if ring.any() else 0.0
    return core_frac, core_px, contrast, skin_frac


def _nms_containment(cands, iou_thr=0.2, contain_thr=0.6):
    """
    Greedy NMS, largest area first. A box is dropped if it overlaps a kept box by
    IoU > iou_thr OR if >contain_thr of the SMALLER box lies inside the kept box.
    (Plain IoU NMS lets a small box nested inside a big one survive.)
    Returns (kept, dropped).
    """
    order = sorted(cands, key=lambda c: c["area"], reverse=True)
    kept, dropped = [], []
    for c in order:
        ax1, ay1, ax2, ay2 = c["draw_bbox"]
        a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        clash = False
        for k in kept:
            bx1, by1, bx2, by2 = k["draw_bbox"]
            iw = min(ax2, bx2) - max(ax1, bx1)
            ih = min(ay2, by2) - max(ay1, by1)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            b_area = max(1, (bx2 - bx1) * (by2 - by1))
            iou = inter / (a_area + b_area - inter)
            if iou > iou_thr or inter / min(a_area, b_area) > contain_thr:
                clash = True
                break
        (dropped if clash else kept).append(c)
    return kept, dropped


def detect_bulb_candidates(image, threshold_value=None, min_area=None, pad_pixels=15,
                           debug=False, rejected=None, max_candidates=40):
    
    h, w = image.shape[:2]
    gray = get_luminosity(image)
    smooth = reduce_haziness(gray, gamma=2.0)

    t = scene_brightness(gray)
    white_thr = int(round(235 - 30 * t))     # 235 night -> 205 day
    min_contrast = 50 - 20 * t               # 50 night  -> 30 day

    if min_area is None:
        min_area = max(4, int(h * w * 0.000004))

    masks = []
    if threshold_value is None:
        peak_luminance = float(np.percentile(smooth, 99.7))
        lo = 175 + int(25 * t)
        hi = 215 + int(15 * t)
        thr_a = min(hi, max(lo, int(peak_luminance * 0.85)))
        masks.append(("A", isolate_active_bulbs(smooth, thr_a), thr_a))

        thr_b = int(np.clip(np.percentile(gray, 99.0), 185 + int(25 * t), 235))
        masks.append(("B", isolate_active_bulbs(gray, thr_b), thr_b))
    else:
        masks.append(("A", isolate_active_bulbs(smooth, threshold_value), threshold_value))

    def reject(box, reason):
        if rejected is not None:
            rejected.append({"draw_bbox": box, "reason": reason})

    raw_candidates = []
    safe_pad = 15 if pad_pixels is None else int(pad_pixels)

    for tag, thresh, thr in masks:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            # mask B is looser, so it needs bigger blobs to count (kills leaf/brick specks)
            if area < (min_area if tag == "A" else max(min_area * 4, 24)):
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            draw_pad = 4
            dx1, dy1 = max(0, bx - draw_pad), max(0, by - draw_pad)
            dx2, dy2 = min(w, bx + bw + draw_pad), min(h, by + bh + draw_pad)

            if not (is_round_or_oval(c) or is_linear_light_source(c)):
                reject([dx1, dy1, dx2, dy2], f"shape[{tag}]")
                continue

            core_frac, core_px, contrast, skin_frac = blob_light_stats(image, gray, c, white_thr)
            if debug:
                print(f"t={t:.2f} mask={tag} thr={thr} area={area:.0f} core={core_frac:.2f} "
                      f"core_px={core_px} contrast={contrast:.0f} skin={skin_frac:.2f}")

            # small clipped core inside a big halo is allowed, but only with strong contrast
            core_ok = core_frac >= 0.10 or (core_px >= 12 and contrast >= min_contrast + 15)
            if not core_ok:
                reject([dx1, dy1, dx2, dy2], f"core[{tag}] {core_frac:.2f}/{core_px}px")
                continue
            if contrast < min_contrast:
                reject([dx1, dy1, dx2, dy2], f"contrast[{tag}] {contrast:.0f}<{min_contrast:.0f}")
                continue

            actual_pad = safe_pad + 10 if max(bw, bh) < 25 else safe_pad
            x1, y1 = max(0, bx - actual_pad), max(0, by - actual_pad)
            x2, y2 = min(w, bx + bw + actual_pad), min(h, by + bh + actual_pad)

            raw_candidates.append({
                "bbox": [x1, y1, x2, y2],
                "draw_bbox": [dx1, dy1, dx2, dy2],
                "area": float(area),
                "core_frac": core_frac,
                "core_px": core_px,
                "contrast": contrast,
                "skin_frac": skin_frac,
                "halo": halo_gradient(gray, c),
                "mask": tag,
            })

    if not raw_candidates:
        return []

    kept, dropped = _nms_containment(raw_candidates)
    for d in dropped:
        reject(d["draw_bbox"], "nms")

    # Cap the VLM workload: keep the most emitter-like candidates
    if len(kept) > max_candidates:
        kept.sort(key=lambda c: c["contrast"] * np.log1p(c["area"]), reverse=True)
        for d in kept[max_candidates:]:
            reject(d["draw_bbox"], "cap")
        kept = kept[:max_candidates]
    return kept