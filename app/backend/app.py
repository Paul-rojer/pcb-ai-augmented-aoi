"""
app.py  (2026-06-21)  -- Roger Vision Labs / AOI PRO backend

FastAPI service that serves the REAL v5 model + calibrated decision fusion to
the web dashboard. CPU-only friendly (no GPU required). Run:

    cd backend
    pip install -r requirements.txt
    python -m uvicorn app:app --host 0.0.0.0 --port 8000

Then the frontend calls these endpoints:

  GET  /health                       -> liveness
  GET  /config                       -> thresholds, categories, pcb types, model info
  POST /inspect   (multipart)        -> run one inspection, returns the decision
        fields: image (file, required)
                pcb_type (str: PCB_1 | PCB_2 | Others; default PCB_1)
                aoi_verdict (str: PASS | FAIL | UNKNOWN; required for Others)
  POST /gradcam   (multipart)        -> diagnostic attention map (PCB_1/PCB_2 only)
        fields: image (file), pcb_type (str)
  GET  /history                      -> recent inspections (for the Inspection Log)
  GET  /stats                        -> aggregates for the Home dashboard charts

A static frontend can be dropped into backend/frontend/ and is served at "/".
"""
import csv
import os
import tempfile
from collections import Counter
from datetime import datetime, timedelta

RETENTION_DAYS = 7   # keep logs/reports for the last 7 days

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import inference

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_CSV = os.path.join(HERE, "inspection_history.csv")
FRONTEND_DIR = os.path.join(HERE, "frontend")

app = FastAPI(title="AOI PRO — Decision Support API", version="1.0")

# Allow the dev frontend (any origin) to call the API. Tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FIELDS = ["timestamp", "pcb_type", "board_label", "mode", "aoi_result",
                  "ai_prediction", "ai_decision", "confidence", "anomaly_score",
                  "lite_score", "fused_score", "decision"]


def _recent(rows):
    """Keep only rows from the last RETENTION_DAYS days."""
    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
    out = []
    for r in rows:
        ts = r.get("timestamp", "")
        try:
            if datetime.fromisoformat(ts) >= cutoff:
                out.append(r)
        except ValueError:
            out.append(r)   # keep rows with unparseable timestamps
    return out


def _migrate_if_needed():
    """If an old CSV with a different header exists, archive it and start fresh."""
    if not os.path.exists(HISTORY_CSV):
        return
    with open(HISTORY_CSV, newline="") as f:
        header = (f.readline().strip().split(","))
    if header != HISTORY_FIELDS:
        try:
            os.replace(HISTORY_CSV, HISTORY_CSV + ".old")
        except OSError:
            pass


def _append_history(row):
    new = not os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})
    _prune_history()


def _prune_history():
    """Drop rows older than the retention window so the file stays bounded."""
    rows = _read_history(prune=False)
    recent = _recent(rows)
    if len(recent) != len(rows):
        with open(HISTORY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            w.writeheader()
            for r in recent:
                w.writerow({k: r.get(k, "") for k in HISTORY_FIELDS})


def _read_history(prune=True):
    if not os.path.exists(HISTORY_CSV):
        return []
    with open(HISTORY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    return _recent(rows) if prune else rows


_migrate_if_needed()


def _save_upload(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


@app.get("/health")
def health():
    return {"status": "ok", "service": "aoi-pro-backend", "version": "1.0"}


@app.get("/config")
def get_config():
    return inference.config()


@app.post("/inspect")
async def inspect(
    image: UploadFile = File(...),
    pcb_type: str = Form("PCB_1"),
    aoi_verdict: str = Form(None),
    engine: str = Form("software"),
):
    path = _save_upload(image)
    try:
        result = inference.inspect(
            path, pcb_type=pcb_type, aoi_verdict=aoi_verdict,
            original_filename=image.filename, engine=engine,
        )
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": str(e), "type": type(e).__name__})
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **result}
    _append_history(row)
    return result


@app.post("/gradcam")
async def gradcam(image: UploadFile = File(...), pcb_type: str = Form("PCB_1")):
    path = _save_upload(image)
    try:
        return inference.gradcam(path, pcb_type)
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": str(e), "type": type(e).__name__})
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.get("/history")
def history(limit: int = 100):
    rows = _read_history()
    return {"count": len(rows), "items": rows[-limit:][::-1]}


@app.get("/stats")
def stats():
    rows = _read_history()
    cats = ["CONFIRMED_PASS", "CONFIRMED_FAIL", "REVIEW",
            "FALSE_REJECT", "MISSED_DEFECT", "UNCERTAIN"]
    by_decision = Counter(r["decision"] for r in rows)
    # confidence histogram (10-wide bins from 50..100)
    bins = {f"{b}-{b+10}": 0 for b in range(50, 100, 10)}
    for r in rows:
        try:
            c = float(r["confidence"])
        except (ValueError, KeyError):
            continue
        b = min(90, max(50, int(c // 10) * 10))
        bins[f"{b}-{b+10}"] = bins.get(f"{b}-{b+10}", 0) + 1
    # volume over time (by date)
    by_date = Counter(r["timestamp"][:10] for r in rows if r.get("timestamp"))
    auto = sum(by_decision.get(c, 0) for c in ("CONFIRMED_PASS", "CONFIRMED_FAIL"))
    return {
        "total": len(rows),
        "auto_resolved": auto,
        "escalated_review": by_decision.get("REVIEW", 0),
        "missed_defect_catches": by_decision.get("MISSED_DEFECT", 0),
        "by_decision": {c: by_decision.get(c, 0) for c in cats},
        "confidence_histogram": bins,
        "volume_by_date": dict(sorted(by_date.items())),
    }


# Serve a static frontend if one has been dropped into backend/frontend/.
if os.path.isdir(FRONTEND_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
