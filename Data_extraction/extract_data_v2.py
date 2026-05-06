"""
Extract texture IDs and LP tensors from DAMASK postResults txt files.

For each sample d_i (i=1..3000):
  - d_i_theta_00deg_inc50.txt → per-element texture (1 value) and LP (9 values) at increment 50

File format (confirmed from d_3_theta_00deg_inc50.txt):
  Line 1  : "2\theader"  → n_header = 2
  Line 2  : command/metadata (skip)
  Line 3  : column names (tab-separated):
      0=inc, 1=elem, 2=node, 3=ip, 4=grain,
      5=1_pos, 6=2_pos, 7=3_pos,
      8..16 = 1_lp..9_lp,
      ...
      53 = texture
  Lines 4+ : data rows (tab-separated, 961 rows for 31x31 grid)

Output:
  texture.npy  shape (3000, 31, 31)   — integer grain/texture ID per element
  lp.npy       shape (3000, 31, 31, 9) — plastic velocity gradient (flattened 3x3)
"""

import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = Path("/home/prabhat/Crystal_plasticity/New/Composite/Isotropic/DP_steel/Elastic_martensite/extract_data")
OUTPUT_DIR = Path("//home/prabhat/Crystal_plasticity/New/Composite/Isotropic/DP_steel/Elastic_martensite/dump_data_500")
N_SAMPLES  = 1000
GRID       = 31           # 31 × 31 elements
N_PIXELS   = GRID * GRID  # 961

# Column indices (0-based) confirmed from header row of the txt files
COL_ELEM    = 1
COL_LP      = slice(8, 17)   # columns 8–16 inclusive → 9 values (1_lp … 9_lp)
COL_P       = slice(44, 53)  # columns 44–53 inclusive → 9 values (1_p … 9_p)
COL_TEXTURE = 53             # texture (grain ID)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_txt(filepath: Path):
    """
    Parse a DAMASK postResults txt file.

    Returns
    -------
    texture_grid : np.ndarray, shape (31, 31)
        Grain/texture ID mapped onto the spatial grid.
    lp_grid : np.ndarray, shape (31, 31, 9)
        Plastic velocity gradient components mapped onto the spatial grid.
    """
    with open(filepath) as f:
        # Line 1: "<n_header>\theader"
        n_header = int(f.readline().split()[0])
        # Skip remaining header lines (n_header - 1 lines)
        for _ in range(n_header - 1):
            f.readline()
        # Next line is the column-name row — skip it
        f.readline()
        # Read all data rows
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]

    data = np.array(rows, dtype=np.float64)  # shape (961, 62)

    elem         = data[:, COL_ELEM].astype(int)  # 1-based element indices
    lp_flat      = data[:, COL_LP]                # (961, 9)
    p_flat       = data[:, COL_P]
    texture_flat = data[:, COL_TEXTURE]          # (961,)

    texture_grid = np.zeros((GRID, GRID),    dtype=np.float64)
    lp_grid      = np.zeros((GRID, GRID, 9), dtype=np.float64)
    p_grid       = np.zeros((GRID, GRID, 9), dtype=np.float64)

    for i, e in enumerate(elem):
        r = (e - 1) // GRID
        c = (e - 1) % GRID
        texture_grid[r, c]  = texture_flat[i]
        lp_grid[r, c]       = lp_flat[i]
        p_grid[r, c]        = p_flat[i]

    return texture_grid, lp_grid, p_grid


# ── Main extraction loop ──────────────────────────────────────────────────────

texture_all = np.zeros((N_SAMPLES, GRID, GRID),    dtype=np.float64)
lp_all      = np.zeros((N_SAMPLES, GRID, GRID, 9), dtype=np.float64)
p_all = np.zeros((N_SAMPLES, GRID, GRID, 9), dtype=np.float64)

missing_txt = []

for i in range(1, N_SAMPLES + 1):
    txt_path = DATA_DIR / f"d_{i}_theta_00deg_inc50.txt"

    if not txt_path.exists():
        missing_txt.append(i)
        continue

    try:
        texture, lp, p     = parse_txt(txt_path)
        texture_all[i - 1] = texture
        lp_all[i - 1]      = lp
        p_all[i - 1]       = p
    except Exception as exc:
        print(f"  [ERROR] sample {i}: {exc}")

    if i % 50 == 0:
        print(f"  Processed {i}/{N_SAMPLES} ...")

# ── Save ──────────────────────────────────────────────────────────────────────
texture_out = OUTPUT_DIR / "texture.npy"
lp_out      = OUTPUT_DIR / "lp.npy"
p_out       = OUTPUT_DIR / "p.npy"


np.save(texture_out, texture_all)
np.save(lp_out,      lp_all)
np.save(p_out, p_all)
print("\n── Summary ──────────────────────────────────────────────────────────")
print(f"texture saved → {texture_out}  shape {texture_all.shape}")
print(f"lp      saved → {lp_out}  shape {lp_all.shape}")
print(f" p      saved → {p_out}   shape {p_all.shape}")

if missing_txt:
    print(f"\nMissing .txt files ({len(missing_txt)}): "
          f"{missing_txt[:10]}{'...' if len(missing_txt) > 10 else ''}")
else:
    print("\nAll .txt files found and processed successfully.")
