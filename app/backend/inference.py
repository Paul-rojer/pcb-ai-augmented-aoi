"""
inference.py  (2026 — Roger Vision Labs / AOI PRO backend)

Golden-template + AOI-consensus inspection pipeline (no torch; OpenCV/NumPy only).

For one image:
  ROI auto-crop -> route to the matching golden board-type template
  -> ORB+ECC registration -> z-score anomaly map -> anomaly_score + match%
  -> AI decision (PASS / REVIEW / FAIL) -> fuse with AOI verdict (_G/_N or operator)
  -> final 6-state decision + heatmap.

API result keys are kept compatible with the existing frontend:
  pcb_type, mode, aoi_result, ai_prediction, confidence (=match%),
  anomaly_score, lite_score, fused_score, decision, board_type, gradcam_available
"""
import base64
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
for _p in (os.path.join(PROJECT, "golden"), os.path.join(PROJECT, "roi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multi_golden import MultiGolden  # noqa: E402

GALLERY = os.path.join(PROJECT, "golden", "gallery")
# auto_fail=True so the AI emits PASS / REVIEW / FAIL (needed for the 6-state fuse)
_mg = MultiGolden(gallery_dir=GALLERY, auto_fail=True)

# Friendly labels for the auto-detected board types.
TYPE_LABELS = {
    "type0": "Board A (blue / ETS SMTMATE)",
    "type1": "Board B (green design A)",
    "type2": "Board C (green design B)",
}


def aoi_from_filename(name):
    """AOI machine code in the filename: _G -> PASS, _N -> FAIL."""
    if not name:
        return None
    stem = os.path.basename(name).rsplit(".", 1)[0].upper()
    if stem.endswith("_G"):
        return "PASS"
    if stem.endswith("_N"):
        return "FAIL"
    return None


def _template(name):
    for t in _mg.templates:
        if t.name == name:
            return t
    return None


def _match_pct(score, t):
    """Convert anomaly score to a 0-100 'match confidence' using the type's tau_fail."""
    if score is None or t is None or not t.tau_fail:
        return None
    pct = 100.0 * (1.0 - score / float(t.tau_fail))
    return round(max(0.0, min(100.0, pct)), 1)


def _fuse(aoi, ai):
    """6-state consensus of AOI verdict + AI decision (PASS/REVIEW/FAIL)."""
    if ai == "REVIEW":
        return "REVIEW"
    if aoi not in ("PASS", "FAIL"):
        return "UNCERTAIN"
    if aoi == "PASS" and ai == "PASS":
        return "CONFIRMED_PASS"
    if aoi == "FAIL" and ai == "FAIL":
        return "CONFIRMED_FAIL"
    if aoi == "FAIL" and ai == "PASS":
        return "FALSE_REJECT"      # AOI rejected, AI says good -> human
    return "MISSED_DEFECT"          # AOI passed, AI flags -> human


def _overlay_png(bgr, zmap):
    if zmap is None:
        return None
    base = cv2.resize(bgr, (zmap.shape[1], zmap.shape[0])) if bgr.shape[:2] != zmap.shape[:2] else bgr
    z = zmap.astype(np.float32)
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    hm = cv2.applyColorMap((z * 255).astype(np.uint8), cv2.COLORMAP_JET)
    ov = cv2.addWeighted(base, 0.6, hm, 0.4, 0)
    ok, buf = cv2.imencode(".png", ov)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def inspect(image_path, pcb_type=None, aoi_verdict=None, original_filename=None,
            engine="software"):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError("Could not read the uploaded image.")

    result, zmap = _mg.inspect(bgr, do_crop="auto", engine=engine)
    board_type = result.get("type")
    score = result.get("score")
    ai = result.get("decision")                      # PASS / REVIEW / FAIL
    t = _template(board_type)
    match = _match_pct(score, t)

    # AOI verdict: filename code first, else operator-supplied
    aoi = aoi_from_filename(original_filename or image_path)
    aoi_source = "filename_code"
    if aoi is None:
        aoi = (aoi_verdict or "UNKNOWN").strip().upper()
        aoi_source = "operator" if aoi_verdict else "unknown"

    decision = _fuse(aoi, ai)
    ai_pred = {"PASS": "GOOD", "FAIL": "DEFECT", "REVIEW": "UNCERTAIN"}.get(ai, "UNCERTAIN")

    return {
        "pcb_type": board_type or "UNKNOWN",
        "board_type": board_type,
        "board_label": TYPE_LABELS.get(board_type, "Unknown board type"),
        "mode": "auto",
        "aoi_result": aoi,
        "aoi_source": aoi_source,
        "ai_prediction": ai_pred,
        "ai_decision": ai,
        "anomaly_score": None if score is None else round(float(score), 3),
        "confidence": match if match is not None else 0.0,   # = match % (kept key for UI)
        "match_pct": match,
        "lite_score": None if score is None else round(float(score), 3),
        "fused_score": None if match is None else round(match / 100.0, 4),
        "decision": decision,
        "gradcam_available": board_type is not None,
        "note": result.get("reason") or None,
        "engine": result.get("engine", engine),
        "hw_metrics": result.get("hw_metrics"),
    }


def gradcam(image_path, pcb_type=None):
    """Anomaly heatmap (our model's z-map) + cropped input as base64 PNGs."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        return {"available": False, "reason": "Could not read image."}
    result, zmap = _mg.inspect(bgr, do_crop="auto")
    if zmap is None:
        return {"available": False,
                "reason": result.get("reason", "Board could not be registered to a known type.")}
    # cropped/registered view = resize of the (auto-cropped) board
    from aoi_profile_crop import crop_image, needs_crop
    view = crop_image(bgr)[0] if needs_crop(bgr) else bgr
    t = _template(result.get("type"))
    ok, cbuf = cv2.imencode(".png", cv2.resize(view, (zmap.shape[1], zmap.shape[0])))
    cropped = "data:image/png;base64," + base64.b64encode(cbuf.tobytes()).decode("ascii") if ok else None
    # border hotspot = peak anomaly near the frame edge
    yx = np.unravel_index(int(np.argmax(zmap)), zmap.shape)
    h, w = zmap.shape
    border = (yx[0] < 0.1 * h or yx[0] > 0.9 * h or yx[1] < 0.1 * w or yx[1] > 0.9 * w)
    return {
        "available": True,
        "overlay_png": _overlay_png(view, zmap),
        "cropped_png": cropped,
        "board_type": result.get("type"),
        "board_label": TYPE_LABELS.get(result.get("type"), ""),
        "anomaly_score": None if result.get("score") is None else round(float(result["score"]), 3),
        "confidence": _match_pct(result.get("score"), t),
        "nearest_component": None,
        "component_distance": None,
        "is_border_hotspot": bool(border),
    }


def config():
    """Static facts for the dashboard."""
    types = [{"id": t.name, "label": TYPE_LABELS.get(t.name, t.name),
              "tau_pass": round(float(t.tau_pass), 3) if t.tau_pass else None,
              "tau_fail": round(float(t.tau_fail), 3) if t.tau_fail else None}
             for t in _mg.templates]
    return {
        "model": "Golden-Template Registration + Change Detection (ECC-aligned, multi-board)",
        "method": "ORB+ECC registration vs per-board golden reference; z-score anomaly; AOI+AI consensus",
        "machine": "ETS-AOI-900",
        "metric": "anomaly_score (0 = perfect match, higher = more anomalous) + match%",
        "auto_board_detection": True,
        "board_types": types,
        "decision_categories": [
            "CONFIRMED_PASS", "CONFIRMED_FAIL", "REVIEW",
            "FALSE_REJECT", "MISSED_DEFECT", "UNCERTAIN",
        ],
        "separation_auc": 0.83,
        "modes": ["manual_upload", "watchdog_folder"],
        "default_mode": "manual_upload",
    }
