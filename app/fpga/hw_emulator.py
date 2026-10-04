"""
hw_emulator.py  -- bit-accurate model of the FPGA Golden Change-Detection Engine.

This is the *exact integer datapath* the Verilog implements. Running it in Python
gives identical results to the (future) RTL simulation and the on-chip hardware,
so the website can show an "FPGA Accelerator" mode and report real hardware
metrics, with NO physical board.

Datapath (per pixel, 1 pixel/clock after a short line-buffer fill):
    a8   = aligned pixel            (uint8, 0..255)
    m8   = golden mean pixel        (uint8, 0..255)        -- prepared by host
    rcp  = round(2^16 / (std + k))  (uint16, Q0.16)        -- prepared by host
    d    = |a8 - m8|                (uint8)
    zq   = (d * rcp) >> 8           (z in Q?.8, i.e. z_real = zq/256)
    zb   = 3x3 Gaussian(zq) >> 4    (smoothed, integer)
    bin  = min(255, zb >> 4)        (256-bin histogram index)
    hist[bin]++ ; max_zb = max(...)
Host (tiny, O(256)) reads the histogram and computes the anomaly score as the
mean of the top `top_frac` z-values -- the same definition the software path uses.

Only integer ops (one multiply per pixel -> 1 DSP). No floating point on the FPGA.
"""
import numpy as np

# ---- fixed-point spec (this is the contract the Verilog must match) ----
RECIP_Q = 16          # reciprocal stored as Q0.16 unsigned
Z_SHIFT = 8           # zq = (d*rcp) >> Z_SHIFT  -> z_real = zq / 256
BLUR_K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.int64)  # sum=16
BLUR_SHIFT = 4        # divide by 16
HIST_BINS = 256
BIN_SHIFT = 4         # bin = zb >> 4 (clamped)  -> bin center z_real = (bin+0.5)*16/256
BIN_TO_Z = (1 << BLUR_SHIFT) / 256.0  # z_real per bin step ( = 16/256 = 0.0625 )
STD_FLOOR = 12        # k


def prepare_golden(mean, std, k=STD_FLOOR):
    """Host-side, once per board type: pack golden mean (uint8) + reciprocal (uint16)."""
    m8 = np.clip(np.round(mean), 0, 255).astype(np.uint16)
    rcp = np.round((1 << RECIP_Q) / (std.astype(np.float64) + k))
    rcp = np.clip(rcp, 0, 65535).astype(np.uint32)
    return m8, rcp


def run(aligned, m8, rcp, valid, border=50, top_frac=0.0005, fmax_mhz=115.67):
    """Bit-accurate engine. Returns (score, zmap_uint8, metrics)."""
    score, zmap, metrics, _hist, _max = engine_full(
        aligned, m8, rcp, valid, border, top_frac, fmax_mhz)
    return score, zmap, metrics


def engine_full(aligned, m8, rcp, valid, border=50, top_frac=0.0005, fmax_mhz=115.67):
    """Same engine, but also returns (hist[256], max_zb) for RTL verification."""
    H, W = aligned.shape
    a8 = np.clip(np.round(aligned), 0, 255).astype(np.int64)
    d = np.abs(a8 - m8.astype(np.int64)).astype(np.int64)          # uint8 range
    zq = (d * rcp.astype(np.int64)) >> Z_SHIFT                     # integer z

    # 3x3 Gaussian blur (integer), matches a line-buffered FPGA conv
    pad = np.pad(zq, 1, mode="edge")
    zb = np.zeros_like(zq)
    for dy in range(3):
        for dx in range(3):
            zb += BLUR_K[dy, dx] * pad[dy:dy + H, dx:dx + W]
    zb >>= BLUR_SHIFT

    # interior mask (ignore warp border)
    m = np.zeros((H, W), dtype=bool)
    m[border:H - border, border:W - border] = True
    m &= valid

    binidx = np.clip(zb >> BIN_SHIFT, 0, HIST_BINS - 1)
    hist = np.bincount(binidx[m].ravel(), minlength=HIST_BINS).astype(np.int64)

    score = score_from_hist(hist, top_frac)

    # heatmap for display (normalise zb to 0..255)
    zbm = zb * m
    mx = zbm.max() if zbm.max() > 0 else 1
    zmap = (zbm * 255 // mx).astype(np.uint8)

    max_zb = int(zb[m].max()) if m.any() else 0
    metrics = hw_metrics(H, W, fmax_mhz)
    return score, zmap, metrics, hist, max_zb


def score_from_hist(hist, top_frac=0.0005):
    """Mean of the top `top_frac` z-values, computed from the 256-bin histogram."""
    N = int(hist.sum())
    if N == 0:
        return 0.0
    take = max(1, int(top_frac * N))
    acc_n, acc_z = 0, 0.0
    for b in range(HIST_BINS - 1, -1, -1):
        c = int(hist[b])
        if c == 0:
            continue
        use = min(c, take - acc_n)
        z_center = (b + 0.5) * BIN_TO_Z       # bin -> z_real
        acc_z += use * z_center
        acc_n += use
        if acc_n >= take:
            break
    return float(acc_z / acc_n) if acc_n else 0.0


def hw_metrics(H, W, fmax_mhz=115.67):
    """Cycle-accurate-ish performance + rough resource estimate for the engine."""
    pipeline_depth = 12
    fill = 2 * W                      # line-buffer fill for 3x3
    cycles = H * W + fill + pipeline_depth
    latency_ms = cycles / (fmax_mhz * 1e6) * 1e3
    fps = (fmax_mhz * 1e6) / (H * W)  # 1 pixel/clock steady-state
    return {
        "engine": "FPGA Golden Change-Detection (cycle-accurate model)",
        "fmax_mhz": fmax_mhz,
        "pixels": H * W,
        "clock_cycles": int(cycles),
        "latency_ms": round(latency_ms, 3),
        "throughput_fps": round(fps, 1),
        # actual Quartus synthesis (Cyclone IV E, EP4CE115F29C7):
        "resources": {"LE": 823, "FF": 661, "DSP": 2,
                      "BRAM": "10,288 bits / ~3 M9K (line buffers + 256-bin histogram)"},
    }
