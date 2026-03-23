"""
generate_loading_conditions.py
===============================
Generates DAMASK 2.x spectral solver .load files for the 6 theta-sampled
principal-frame loading conditions used to calibrate MKS plastic localization
kernels (Yabansu & Kalidindi framework).

Setup:
  - Grid      : 31 x 31 x 1  (2D, one layer in Z)
  - |<Dp>|    : 0.02
  - Theta grid: 0, 10, 20, 30, 40, 50 degrees  (from [0, 60))
  - No spin   : <W> = 0
  - Load time : 0.5 s, 50 increments  (adjust to taste)

θ-parameterization (traceless unit tensor in principal frame):
  d11(θ) = sqrt(2/3) * cos(θ -π/3)
  d22(θ) = sqrt(2/3) * cos(θ + π/3)
  d33(θ) = sqrt(2/3) * -cos(θ)
  → trace = 0, ||d^p|| = 1  by construction

Fdot ≈ Lp = Dp  (pure plastic, no spin, no elastic)
"""

import numpy as np
import os

# ── Parameters ────────────────────────────────────────────────────────────────
DP_MAG      = 0.02          # |<Dp>|
THETA_DEG   = [0, 10, 20, 30, 40, 50]   # θ sampling in [0, 60)
LOAD_TIME   = 0.5           # total time per loadcase  [s]
N_INCREMENTS = 50           # number of increments
OUTPUT_DIR  = "./load_files"

# ── θ → Fdot diagonal ─────────────────────────────────────────────────────────
def theta_to_dp_principal(theta_deg, dp_mag=DP_MAG):
    """
    Returns (d11, d22, d33) for the macroscopic plastic stretching tensor
    expressed in its principal frame, for a given θ (degrees).

    The tensor is traceless and ||d|| = dp_mag.
    """
    theta = np.radians(theta_deg)
    c     = np.sqrt(2.0 / 3.0)
    d11   = c * np.cos(theta - np.pi/3.0)                   * dp_mag
    d22   = c * np.cos(theta + np.pi/3.0)  * dp_mag
    d33   = -c * np.cos(theta)  * dp_mag
    return d11, d22, d33

# ── Verify tracelessness and unit magnitude ───────────────────────────────────
def verify_dp(theta_deg):
    d11, d22, d33 = theta_to_dp_principal(theta_deg, dp_mag=1.0)   # unit mag
    trace = d11 + d22 + d33
    norm  = np.sqrt(d11**2 + d22**2 + d33**2)
    return trace, norm

print("=" * 60)
print(f"  θ-parameterization check  (|Dp|=1 for verification)")
print("=" * 60)
print(f"{'θ (°)':>8}  {'d11':>10}  {'d22':>10}  {'d33':>10}  {'trace':>8}  {'||d||':>8}")
print("-" * 62)
for θ in THETA_DEG:
    d11, d22, d33 = theta_to_dp_principal(θ, dp_mag=1.0)
    tr  = d11 + d22 + d33
    nm  = np.sqrt(d11**2 + d22**2 + d33**2)
    print(f"{θ:>8}  {d11:>10.5f}  {d22:>10.5f}  {d33:>10.5f}  {tr:>8.2e}  {nm:>8.5f}")

# ── DAMASK 2.x load file writer ───────────────────────────────────────────────
LOAD_TEMPLATE = """\
# DAMASK spectral load file
# Theta = {theta_deg:.1f} deg
# |<Dp>| = {dp_mag:.4f}

fdot {d11:.6e} 0 0  0 {d22:.6e} 0  0 0 {d33:.6e}   time {t:.1f}   incs {f:d}   freq 1
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n")
print("=" * 60)
print(f"  Writing .load files → {OUTPUT_DIR}/")
print("=" * 60)
print(f"{'File':>30}  {'d11':>12}  {'d22':>12}  {'d33':>12}")
print("-" * 72)

load_data = []   # store for downstream use

for θ in THETA_DEG:
    d11, d22, d33 = theta_to_dp_principal(θ, dp_mag=DP_MAG)

    content = LOAD_TEMPLATE.format(
        theta_deg = θ,
        dp_mag    = DP_MAG,
        d11 = d11, d22 = d22, d33 = d33,
        t = LOAD_TIME,
        f = N_INCREMENTS,
    )

    fname = f"theta_{θ:02d}deg.load"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w") as fp:
        fp.write(content)

    load_data.append({"theta_deg": θ, "d11": d11, "d22": d22, "d33": d33,
                      "file": fpath})
    print(f"{fname:>30}  {d11:>12.6e}  {d22:>12.6e}  {d33:>12.6e}")

# ── Also write a summary table as numpy archive ───────────────────────────────
summary_path = os.path.join(OUTPUT_DIR, "loading_summary.npz")
np.savez(summary_path,
         theta_deg   = np.array([d["theta_deg"] for d in load_data]),
         d11         = np.array([d["d11"]       for d in load_data]),
         d22         = np.array([d["d22"]       for d in load_data]),
         d33         = np.array([d["d33"]       for d in load_data]),
         dp_mag      = np.array([DP_MAG]),
)

print(f"\n  Summary saved → {summary_path}")

# ── Print ready-to-paste Fdot table ───────────────────────────────────────────
print("\n")
print("=" * 60)
print("  Copy-paste Fdot values  (|<Dp>| = {:.4f})".format(DP_MAG))
print("=" * 60)
for d in load_data:
    θ = d["theta_deg"]
    print(f"\n  θ = {θ:>2d}°")
    D = np.diag([d["d11"], d["d22"], d["d33"]])
    for row in D:
        print("    " + "  ".join(f"{v:+.6e}" for v in row))

print("\nDone. Run each .load file with:")
print("  DAMASK_spectral --load theta_00deg.load --geom your_microstructure.geom")
