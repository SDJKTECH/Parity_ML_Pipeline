from __future__ import annotations
import math
import numpy as np
from Pipeline.config import MAX_DRIFT_PIXELS
from Pipeline.context import PipelineContext

# Small movable / stacked items that should not trigger rigid spatial drift
CONSUMABLE_OR_STACKABLE = {
    "cup", "bottle", "bowl", "wine glass", "fork", "knife", "spoon"
}

def _calculate_box_iou(boxA: list[float], boxB: list[float]) -> float:
    """Calculates Intersection over Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = boxA_area + boxB_area - inter_area
    return inter_area / union if union > 0 else 0.0


def _is_edge_or_border_object(bbox: list[float], img_w: int, img_h: int, valid_mask: np.ndarray | None, border_margin: int = 15) -> bool:
    """Detects whether an object sits on the frame boundary or touches the unwarped black border."""
    x1, y1, x2, y2 = bbox
    # 1. Check if the box touches the extreme outer canvas boundaries
    if x1 <= border_margin or y1 <= border_margin or x2 >= (img_w - border_margin) or y2 >= (img_h - border_margin):
        return True

    # 2. Check if the object intersects unwarped invalid mask pixels
    if valid_mask is not None:
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(img_w, int(x2)), min(img_h, int(y2))
        if ix2 > ix1 and iy2 > iy1:
            patch = valid_mask[iy1:iy2, ix1:ix2]
            valid_ratio = np.count_nonzero(patch) / patch.size if patch.size > 0 else 0.0
            if valid_ratio < 0.70:
                return True
    return False


def run(ctx: PipelineContext) -> bool:
    """STEP 4: Mode-Aware Delta Calculations with Edge-Filtering and Stack-Tolerance."""
    checklist = []
    mode = getattr(ctx, "audit_mode", "ALL")
    print(f"\n--- Reset Checklist ({mode} Mode) ---")

    # =========================================================================
    # BRANCH A: OBJECTS & FURNITURE DRIFT (Only runs for 'OBJECTS' or 'ALL')
    # =========================================================================
    if mode in ("OBJECTS", "ALL"):
        master_objs = ctx.master_data.get("objects", [])
        current_objs = ctx.current_data.get("objects", []).copy()

        # Canvas bounds for border detection
        canvas_h, canvas_w = 480, 640
        if ctx.aligned_current_img is not None:
            canvas_h, canvas_w = ctx.aligned_current_img.shape[:2]
        elif ctx.raw_current_img is not None:
            canvas_h, canvas_w = ctx.raw_current_img.shape[:2]

        matched_master = []
        ctx.drift_alerts = []

        for m_obj in master_objs:
            best_match_idx = None
            min_dist = float("inf")
            best_iou = 0.0

            for idx, c_obj in enumerate(current_objs):
                if m_obj["label"] == c_obj["label"]:
                    dist = math.hypot(
                        m_obj["centroid"][0] - c_obj["centroid"][0],
                        m_obj["centroid"][1] - c_obj["centroid"][1],
                    )
                    iou = _calculate_box_iou(m_obj["bbox"], c_obj["bbox"])
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = idx
                        best_iou = iou

            if best_match_idx is not None:
                matched_master.append(m_obj)
                matched_c_obj = current_objs.pop(best_match_idx)

                # Small consumables & stacked items skip rigid spatial drift
                is_consumable = m_obj["label"].lower() in CONSUMABLE_OR_STACKABLE

                # Allow perspective & IoU tolerance
                if not is_consumable:
                    if min_dist > MAX_DRIFT_PIXELS and best_iou < 0.35:
                        ctx.drift_alerts.append(
                            f"{m_obj['label'].capitalize()} shifted {int(min_dist)}px"
                        )

        # Missing items calculation (Ignore minor count discrepancies on consumables if some are present)
        ctx.missing_items = []
        for obj in master_objs:
            if obj not in matched_master:
                label = obj["label"]
                # If it's a cup/bottle and we still have other cups in the scene, don't flag missing
                if label.lower() in CONSUMABLE_OR_STACKABLE:
                    any_present = any(m["label"] == label for m in matched_master)
                    if any_present:
                        continue
                ctx.missing_items.append(label)

        # Filter out boundary/clipping artifacts from clutter list
        filtered_clutter = []
        for obj in current_objs:
            if _is_edge_or_border_object(obj["bbox"], canvas_w, canvas_h, ctx.valid_mask):
                continue
            # Also ignore single extraneous paper cups as critical clutter
            if obj["label"].lower() in CONSUMABLE_OR_STACKABLE:
                continue
            filtered_clutter.append(obj["label"])

        ctx.clutter_items = filtered_clutter

        if not ctx.missing_items:
            checklist.append("[OK] All required room items present.")
        else:
            for item in set(ctx.missing_items):
                msg = f"[MISSING] {ctx.missing_items.count(item)} x {item}"
                checklist.append(msg)
                print(msg)

        if not ctx.drift_alerts:
            checklist.append("[OK] Furniture positions verified.")
        else:
            for alert in ctx.drift_alerts:
                msg = f"[DRIFT] {alert}"
                checklist.append(msg)
                print(msg)

        if not ctx.clutter_items:
            checklist.append("[OK] Room is clean (No clutter detected).")
        else:
            for item in set(ctx.clutter_items):
                msg = f"[CLUTTER] Remove {ctx.clutter_items.count(item)} x {item}"
                checklist.append(msg)
                print(msg)

    # =========================================================================
    # BRANCH B: BULB AUDIT (Strictly Untouched)
    # =========================================================================
    if mode in ("BULBS", "ALL"):
        expected_bulbs = ctx.master_data.get(
            "bulb_detections"
        ) or ctx.master_data.get("bulbs", [])
        valid_mask = getattr(ctx, "valid_mask", None)

        out_of_view_bulbs = []
        if valid_mask is not None and expected_bulbs:
            h, w = valid_mask.shape[:2]
            master_shape = ctx.master_data.get("image_shape")
            scale_x, scale_y = 1.0, 1.0
            if master_shape and len(master_shape) >= 2:
                orig_h, orig_w = master_shape[0], master_shape[1]
                if orig_w > 0 and orig_h > 0:
                    scale_x = w / float(orig_w)
                    scale_y = h / float(orig_h)

            for b_idx, m_bulb in enumerate(expected_bulbs, start=1):
                m_box = m_bulb.get("bbox", [])
                if len(m_box) == 4:
                    mx1 = int(m_box[0] * scale_x)
                    my1 = int(m_box[1] * scale_y)
                    mx2 = int(m_box[2] * scale_x)
                    my2 = int(m_box[3] * scale_y)
                    mx1, my1 = max(0, min(w - 1, mx1)), max(0, min(h - 1, my1))
                    mx2, my2 = max(0, min(w - 1, mx2)), max(0, min(h - 1, my2))
                    if mx2 > mx1 and my2 > my1:
                        patch = valid_mask[my1:my2, mx1:mx2]
                        vis_ratio = (
                            np.count_nonzero(patch) / patch.size if patch.size > 0 else 0
                        )
                        if vis_ratio < 0.35:
                            out_of_view_bulbs.append(b_idx)

        ctx.out_of_view_bulbs = out_of_view_bulbs
        visible_expected = max(0, len(expected_bulbs) - len(out_of_view_bulbs))
        current_bulbs = getattr(ctx, "current_bulb_count", 0)

        if out_of_view_bulbs:
            bulb_tags = ", #".join(str(b) for b in out_of_view_bulbs)
            msg = (
                f"[OUT OF VIEW] {len(out_of_view_bulbs)} bulb(s) outside camera angle"
                f" (Bulb #{bulb_tags})."
            )
            checklist.append(msg)
            print(msg)

        if current_bulbs >= visible_expected and visible_expected > 0:
            msg = (
                f"[OK] Bulb status verified: All {current_bulbs} visible light(s) are active."
            )
            checklist.append(msg)
            print(msg)
        elif current_bulbs < visible_expected:
            turned_off = visible_expected - current_bulbs
            msg = f"[LIGHTS OFF] {turned_off} visible light source(s) appear inactive or turned off."
            checklist.append(msg)
            print(msg)
        elif visible_expected == 0:
            msg = "[OK] No active light sources expected in this view."
            checklist.append(msg)
            print(msg)

    ctx.reset_checklist = checklist

    # =========================================================================
    # VERDICT EVALUATION
    # =========================================================================
    if mode == "BULBS":
        visible_expected = max(
            0,
            len(ctx.master_data.get("bulb_detections", []))
            - len(getattr(ctx, "out_of_view_bulbs", [])),
        )
        is_rejected = getattr(ctx, "current_bulb_count", 0) < visible_expected
    elif mode == "OBJECTS":
        is_rejected = (len(ctx.missing_items) > 0) or (len(ctx.drift_alerts) > 0) or (len(ctx.clutter_items) > 0)
    else:
        visible_expected = max(
            0,
            len(ctx.master_data.get("bulb_detections", []))
            - len(getattr(ctx, "out_of_view_bulbs", [])),
        )
        is_rejected = (
            getattr(ctx, "current_bulb_count", 0) < visible_expected
        ) or (len(ctx.missing_items) > 0) or (len(ctx.clutter_items) > 0)

    ctx.verdict = "REJECTED" if is_rejected else "OK"
    return True