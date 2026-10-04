# AOI PRO — Backend (FastAPI)

Serves the **real v5 model + calibrated decision fusion** to the web dashboard.
Does not train or modify any model; it only loads `model_v5/best_model_v5.pth`
and the existing calibrations. CPU-only friendly — no GPU required, so it runs
on any AOI PC.

## Run

```bat
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

(or just double-click `run_backend.bat` on Windows). Open
`http://localhost:8000/health` to confirm it's up, and
`http://localhost:8000/docs` for the interactive API explorer.

## API contract (what the frontend calls)

### `GET /health`
`{ "status": "ok", "service": "aoi-pro-backend", "version": "1.0" }`

### `GET /config`
Thresholds, decision categories, PCB types, model info, and the recovery-gate
calibration — for populating dashboard labels and the comparison panel.

### `POST /inspect`  (multipart/form-data)
| field        | type   | notes |
|--------------|--------|-------|
| `image`      | file   | required — the board photo |
| `pcb_type`   | string | `PCB_1` \| `PCB_2` \| `Others` (default `PCB_1`) |
| `aoi_verdict`| string | `PASS` \| `FAIL` \| `UNKNOWN` — **required for `Others`**; for PCB_1/PCB_2 it is read from the image filename code (`_G`/`_N`) automatically |

Response:
```json
{
  "pcb_type": "PCB_1",
  "mode": "supported",
  "aoi_result": "PASS",
  "aoi_source": "filename_code",
  "ai_prediction": "defective",
  "confidence": 65.66,
  "lite_score": 0.018631,
  "fused_score": -1.3812,
  "decision": "CONFIRMED_PASS",
  "gradcam_available": true,
  "note": null
}
```
`decision` is one of the six outcomes. Every inspection is appended to
`inspection_history.csv`.

### `POST /gradcam`  (multipart/form-data)
Diagnostic attention map (PCB_1/PCB_2 only; returns `available: false` for
Others). **Never** influences the decision.

### `GET /history?limit=100`
Recent inspections (newest first) for the Inspection Log page.

### `GET /stats`
Aggregates for the Home dashboard charts: totals, counts per decision category,
confidence histogram, and volume-by-date.

## Serving the frontend

Drop your exported UI (from Stitch / AI Studio) into `backend/frontend/` and it
will be served at `http://localhost:8000/`. Point the UI's API calls at the same
origin (e.g. `/inspect`). For a single self-contained AOI-PC deployment, this is
all you need: one Python process serving both the model API and the UI.

## Notes
- The CNN (`predict_single_v5`) needs `torch`; it is imported lazily, so
  `/config`, `/health`, `/stats` and the PatchCore-lite scoring work even before
  torch is installed.
- Nothing here touches `model_v4_final/` or retrains anything.
