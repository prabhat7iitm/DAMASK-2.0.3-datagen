"""
generate_delta_geom.py
=======================
Generates delta microstructure .geom files in the exact DAMASK 2.x format
matching d_2.geom (self-contained: microstructure + texture inside header).

Delta microstructure layout (31x31x1):
  Grain 1  = inclusion, one voxel at center (16,16) in 1-indexed
  Grain 2  = matrix, all remaining 960 voxels

Each .geom file contains its own Euler angles — no material.config needed.
Euler angles are sampled in DEGREES from the full Bunge space
(phi1 in [0,360), Phi in [0,180], phi2 in [0,360)).

Usage:
  python generate_delta_geom.py               # generates 200 files
  python generate_delta_geom.py --n 50        # generates 50 files
  python generate_delta_geom.py --n 200 --outdir ./my_delta_dir
"""

import numpy as np
import os
import argparse

# ── Configuration ─────────────────────────────────────────────────────────────
GRID_X      = 31
GRID_Y      = 31
GRID_Z      = 1
N_PAIRS     = 200
RANDOM_SEED = 99
OUTPUT_DIR  = "./delta_geom"
SIZE_X      = 1.0
SIZE_Y      = 1.0
SIZE_Z      = 1.0 / GRID_X   # = 0.03225806451612903

# Inclusion at center (1-indexed for DAMASK)
CX = GRID_X // 2 + 1   # = 16
CY = GRID_Y // 2 + 1   # = 16


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ORIENTATION SAMPLING  (full Bunge space, degrees)
# ══════════════════════════════════════════════════════════════════════════════

def sample_orientations(n: int, rng: np.random.Generator,
                        phi_min_deg: float = 1.0) -> np.ndarray:
    """
    Sample n Bunge-Euler angles in DEGREES.
    phi1 in [0, 360)
    Phi  in [phi_min_deg, 180]  (avoid GSH singularity at Phi=0)
    phi2 in [0, 360)
    Returns (n, 3) float array.
    """
    phi1 = rng.uniform(0.0,   360.0, n)
    cos_max = np.cos(np.radians(phi_min_deg))
    cos_min = np.cos(np.radians(180.0))
    Phi  = np.degrees(np.arccos(rng.uniform(cos_min, cos_max, n)))
    phi2 = rng.uniform(0.0,   360.0, n)
    return np.column_stack([phi1, Phi, phi2])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BUILD VOXEL GRID  (31 rows × 31 values, grain 2 everywhere, 1 at center)
# ══════════════════════════════════════════════════════════════════════════════

def build_voxel_lines(cx: int = CX, cy: int = CY,
                      gx: int = GRID_X, gy: int = GRID_Y) -> list:
    """
    Build the voxel data block as a list of strings.
    Each string is one row (b index), with a varying 1..gx.
    Grain 1 at (a=cx, b=cy), grain 2 everywhere else.
    Values are space-separated, right-aligned in 2-char fields.
    """
    lines = []
    for b in range(1, gy + 1):
        row = []
        for a in range(1, gx + 1):
            if a == cx and b == cy:
                row.append(1)
            else:
                row.append(2)
        # Match format: values right-aligned in width 2, space-separated
        lines.append(" ".join(f"{v:2d}" for v in row))
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# 3.  WRITE ONE .GEOM FILE
# ══════════════════════════════════════════════════════════════════════════════

def write_delta_geom(path: str,
                     euler_alpha_deg: np.ndarray,
                     euler_beta_deg:  np.ndarray) -> None:
    """
    Write a single self-contained delta .geom file.

    euler_alpha_deg : (3,) [phi1, Phi, phi2] of inclusion (Grain01), degrees
    euler_beta_deg  : (3,) [phi1, Phi, phi2] of matrix    (Grain02), degrees
    """
    phi1_a, Phi_a, phi2_a = euler_alpha_deg
    phi1_b, Phi_b, phi2_b = euler_beta_deg

    # ── Build header lines (everything before <!skip>) ────────────────────
    header_lines = [
        f"geom_fromVoronoiTessellation v2.0.3 -g {GRID_X} {GRID_Y} {GRID_Z}",
        f"grid\ta {GRID_X}\tb {GRID_Y}\tc {GRID_Z}",
        f"size\tx {SIZE_X}\ty {SIZE_Y}\tz {SIZE_Z:.17g}",
        f"origin\tx 0.0\ty 0.0\tz 0.0",
        f"homogenization\t1",
        f"microstructures\t2",
        f"<microstructure>",
        f"[Grain01]",
        f"crystallite 1",
        f"(constituent)\tphase 1\ttexture  1\tfraction 1.0",
        f"[Grain02]",
        f"crystallite 1",
        f"(constituent)\tphase 1\ttexture  2\tfraction 1.0",
        f"<texture>",
        f"[Grain01]",
        f"(gauss)\tphi1 {phi1_a:.4f}\tPhi {Phi_a:.4f}\tphi2 {phi2_a:.4f}\tscatter 0.0\tfraction 1.0",
        f"[Grain02]",
        f"(gauss)\tphi1 {phi1_b:.4f}\tPhi {Phi_b:.4f}\tphi2 {phi2_b:.4f}\tscatter 0.0\tfraction 1.0",
        f"<!skip>",
    ]

    n_header = len(header_lines)   # 19 for 2-grain delta

    voxel_lines = build_voxel_lines()

    with open(path, "w") as fp:
        # First line: header count
        fp.write(f"{n_header}\theader\n")
        # Header block
        for line in header_lines:
            fp.write(line + "\n")
        # Voxel data
        for line in voxel_lines:
            fp.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN GENERATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def generate_all(n_pairs:    int = N_PAIRS,
                 output_dir: str = OUTPUT_DIR,
                 seed:       int = RANDOM_SEED) -> dict:
    """
    Generate n_pairs delta .geom files.
    Returns dict with euler_alpha and euler_beta arrays.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    euler_alpha = sample_orientations(n_pairs, rng)   # (n_pairs, 3) degrees
    euler_beta  = sample_orientations(n_pairs, rng)   # (n_pairs, 3) degrees

    print(f"  Writing {n_pairs} delta .geom files → {output_dir}/")
    for i in range(n_pairs):
        path = os.path.join(output_dir, f"delta_{i}.geom")
        write_delta_geom(path, euler_alpha[i], euler_beta[i])
        if (i + 1) % 50 == 0 or (i + 1) == n_pairs:
            print(f"    {i+1}/{n_pairs}  ({path})")

    # Save orientation catalogue
    cat_path = os.path.join(output_dir, "delta_catalogue.npz")
    np.savez_compressed(cat_path,
        euler_alpha_deg = euler_alpha,
        euler_beta_deg  = euler_beta,
        n_pairs         = np.array([n_pairs]),
        grid            = np.array([GRID_X, GRID_Y]),
        center_voxel    = np.array([CX, CY]),
    )
    print(f"\n  Catalogue saved → {cat_path}")
    print(f"  euler_alpha (inclusion) : {euler_alpha.shape}  degrees")
    print(f"  euler_beta  (matrix)    : {euler_beta.shape}  degrees")

    return {"euler_alpha": euler_alpha, "euler_beta": euler_beta}


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VERIFICATION  (compare output with d_2.geom format)
# ══════════════════════════════════════════════════════════════════════════════

def verify_output(sample_path: str):
    """Print the first file for visual inspection."""
    print(f"\n  First 25 lines of {sample_path}:")
    with open(sample_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines[:25]):
        print(f"  {i+1:>4d}  {line}", end="")

    # Count grains
    voxels = []
    in_data = False
    header_n = int(lines[0].split()[0])
    for line in lines[header_n + 1:]:
        voxels.extend(int(x) for x in line.split())
    arr = np.array(voxels)
    print(f"\n  Grain 1 count: {(arr==1).sum()}  (should be 1)")
    print(f"  Grain 2 count: {(arr==2).sum()}  (should be 960)")
    print(f"  Total voxels : {len(arr)}  (should be 961)")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate self-contained delta .geom files for MKS"
    )
    parser.add_argument("--n",       type=int, default=N_PAIRS,
                        help=f"Number of orientation pairs (default {N_PAIRS})")
    parser.add_argument("--outdir",  default=OUTPUT_DIR)
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED)
    parser.add_argument("--verify",  action="store_true",
                        help="Print first file for inspection after generating")
    args = parser.parse_args()

    print("=" * 62)
    print(f"  Delta .geom generator")
    print(f"  Grid     : {GRID_X}×{GRID_Y}×{GRID_Z}")
    print(f"  Center   : a={CX}  b={CY}  (1-indexed, Grain01)")
    print(f"  N pairs  : {args.n}")
    print(f"  Output   : {args.outdir}")
    print("=" * 62)

    data = generate_all(args.n, args.outdir, args.seed)

    if args.verify:
        first = os.path.join(args.outdir, "delta_1.geom")
        verify_output(first)

    print(f"\n  Run DAMASK for each .geom file:")
    print(f"    DAMASK_spectral --load theta_00deg.load --geom delta_0001.geom")
    print(f"    ... repeat for all {args.n} files × 6 theta = {args.n*6} total runs")
