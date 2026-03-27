"""
Extract Euler angles and LP tensors from DAMASK simulation files.

For each sample d_i (i=1..500):
  - d_i.geom        → grain map (31x31) + per-grain Euler angles (phi1, Phi, phi2)
  - d_i_theta_00deg_inc50.txt → per-element LP (9 values) at increment 50

Output:
  euler_angles.npy  shape (500, 31, 31, 3)  — Bunge Euler angles in degrees
  lp.npy            shape (500, 31, 31, 9)  — plastic velocity gradient (flattened 3x3)
"""

import os
import re
import numpy as np
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = Path("/home/prabhat/Crystal_plasticity/New/Poly/Ni/FZ_loading/00/extract_data")   # folder containing the files
OUTPUT_DIR = Path("/home/prabhat/Crystal_plasticity/New/Poly/Ni/FZ_loading/00/dump_data_new")
N_SAMPLES  = 2000
GRID       = 31          # 31 x 31 pixels
N_PIXELS   = GRID * GRID # 961

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_geom(filepath: Path):
    """Return (euler_map, grain_grid).

    euler_map : dict  grain_id (1-based) → (phi1, Phi, phi2) in degrees
    grain_grid: ndarray (31, 31) int  — grain ID per pixel
    """
    with open(filepath) as f:
        n_header = int(f.readline().split()[0])
        header_lines = [f.readline() for _ in range(n_header - 1)]
        content = "".join(header_lines)

        # Euler angles per grain from <texture> section
        pattern = r"\(gauss\)\tphi1\s+([\d.]+)\tPhi\s+([\d.]+)\tphi2\s+([\d.]+)"
        matches = re.findall(pattern, content)
        euler_map = {i + 1: (float(p1), float(P), float(p2))
                     for i, (p1, P, p2) in enumerate(matches)}

        # Grid data (after <!skip> marker)
        remaining = f.read()

    lines = remaining.strip().split("\n")
    skip_idx = next(i for i, l in enumerate(lines) if "<!skip>" in l)
    grid_lines = lines[skip_idx + 1:]

    grain_grid = np.array(
        [[int(x) for x in row.split()] for row in grid_lines],
        dtype=np.int32
    )  # shape (31, 31)
    return euler_map, grain_grid


def euler_grid_from_geom(euler_map, grain_grid):
    """Map per-grain Euler angles onto the pixel grid → shape (31, 31, 3)."""
    out = np.zeros((GRID, GRID, 3), dtype=np.float64)
    for r in range(GRID):
        for c in range(GRID):
            gid = grain_grid[r, c]
            out[r, c] = euler_map[gid]
    return out


def parse_txt_lp(filepath: Path):
    """Return LP array shaped (31, 31, 9) from the postResults txt file."""
    with open(filepath) as f:
        n_header = int(f.readline().split()[0])
        for _ in range(n_header - 1):
            f.readline()
        _col_line = f.readline()          # column-name row (skip)
        rows = [line.strip().split("\t") for line in f if line.strip()]

    data = np.array(rows, dtype=np.float64)  # (961, 62)

    # Columns 8–16 (0-indexed, inclusive) → lp_1 … lp_9
    lp_flat = data[:, 8:17]              # (961, 9)
    elem    = data[:, 1].astype(int)     # 1-based element index

    lp_grid = np.zeros((GRID, GRID, 9), dtype=np.float64)
    for i, e in enumerate(elem):
        r = (e - 1) // GRID
        c = (e - 1) % GRID
        lp_grid[r, c] = lp_flat[i]

    return lp_grid


# ── Main extraction loop ──────────────────────────────────────────────────────

euler_all = np.zeros((N_SAMPLES, GRID, GRID, 3), dtype=np.float64)
lp_all    = np.zeros((N_SAMPLES, GRID, GRID, 9), dtype=np.float64)

missing_geom = []
missing_txt  = []

for i in range(1, N_SAMPLES + 1):
    geom_path = DATA_DIR / f"d_{i}.geom"
    txt_path  = DATA_DIR / f"d_{i}_theta_00deg_inc50.txt"

    geom_ok = geom_path.exists()
    txt_ok  = txt_path.exists()

    if not geom_ok:
        missing_geom.append(i)
    if not txt_ok:
        missing_txt.append(i)

    if geom_ok and txt_ok:
        try:
            euler_map, grain_grid = parse_geom(geom_path)
            euler_all[i - 1]     = euler_grid_from_geom(euler_map, grain_grid)
            lp_all[i - 1]        = parse_txt_lp(txt_path)
        except Exception as exc:
            print(f"  [ERROR] sample {i}: {exc}")
    
    if i % 50 == 0:
        print(f"  Processed {i}/{N_SAMPLES} ...")

# ── Save ──────────────────────────────────────────────────────────────────────
euler_out = OUTPUT_DIR / "euler_angles.npy"
lp_out    = OUTPUT_DIR / "lp.npy"

np.save(euler_out, euler_all)
np.save(lp_out,    lp_all)

print("\n── Summary ──────────────────────────────────────────────────")
print(f"euler_angles saved → {euler_out}  shape {euler_all.shape}")
print(f"lp           saved → {lp_out}  shape {lp_all.shape}")

if missing_geom:
    print(f"\nMissing .geom files ({len(missing_geom)}): {missing_geom[:10]}{'...' if len(missing_geom)>10 else ''}")
if missing_txt:
    print(f"Missing .txt  files ({len(missing_txt)}): {missing_txt[:10]}{'...' if len(missing_txt)>10 else ''}")

# ── Quick sanity check ────────────────────────────────────────────────────────
print("\n── Sanity check (sample 0 / d_1) ───────────────────────────")
print(f"Euler [0,0,0] pixel:  {euler_all[0, 0, 0]}  (phi1, Phi, phi2 deg)")
print(f"LP    [0,0,0] pixel:  {lp_all[0, 0, 0]}")
print(f"Euler range: min={euler_all[0].min():.3f}  max={euler_all[0].max():.3f}")
print(f"LP    range: min={lp_all[0].min():.6f}  max={lp_all[0].max():.6f}")
