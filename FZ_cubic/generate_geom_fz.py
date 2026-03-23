"""
generate_geom_fz.py  (FIXED — paper FZ definition)
====================================================
Generate N DAMASK .geom files where grain orientations are uniformly
distributed over the cubic FZ as defined in the paper:

  FZ_C = { g = (φ₁, Φ, φ₂) | 0 ≤ φ₁ < 2π,
            cos⁻¹( cos φ₂ / √(1 + cos² φ₂) ) ≤ Φ ≤ π/2,
            0 ≤ φ₂ ≤ π/4 }

Key properties of this FZ
--------------------------
  φ₂ ∈ [0°,  45°]   — always
  Φ  ∈ [45°, 90°]   — approximately (lower bound depends on φ₂)
  Φ_min(φ₂=0°)  = arccos(1/√2)  = 45.00°
  Φ_min(φ₂=45°) = arccos(1/√3)  = 54.74°  ← the "54.74° limit" is the LOWER bound at φ₂=45°

Previous bugs (WRONG)
---------------------
  1. φ₂ sampled from [0°, 45°]  ← range correct but weighting wrong
  2. Φ  sampled from [0°, 45°]  ← completely wrong: FZ requires Φ ≥ 45°
  3. No rejection mask applied  ← FZ boundary never enforced
  Effect: all grains had Φ < 45° (below FZ minimum) → wrong orientation space

Correct sampling strategy (this file)
--------------------------------------
  Exact inverse-CDF sampling — no rejection needed:

  Step 1: φ₁ ~ Uniform(0, 2π)

  Step 2: φ₂ ~ weighted by ∫ sin(Φ) dΦ from Φ_min(φ₂) to π/2
                           = cos(Φ_min(φ₂)) = cos(φ₂)/√(1+cos²(φ₂))
          Use rejection from Uniform(0, π/4):
            accept prob = cos(φ₂)/√(1+cos²(φ₂))  /  (1/√2)
            range: 1.0 at φ₂=0 to 0.816 at φ₂=45° → ~91% acceptance

  Step 3: Given φ₂, sample Φ ∝ sin(Φ) in [Φ_min(φ₂), π/2]:
            Inverse CDF: Φ = arccos( cos(Φ_min) × U ),  U ~ Uniform(0,1)
            (uses cos(π/2) = 0 to simplify expression)

Usage
-----
  python generate_geom_fz.py --n 2000 --outdir ./geom_fz --seed 42
"""

import numpy as np
import os
import argparse


# ── FZ boundary ───────────────────────────────────────────────────────────────

def phi_min(phi2_rad: np.ndarray) -> np.ndarray:
    """
    Lower bound on Phi for given phi2 (both in radians).

    Φ_min(φ₂) = arccos( cos(φ₂) / √(1 + cos²(φ₂)) )

    At φ₂=0:    Φ_min = arccos(1/√2) = 45.00°
    At φ₂=π/4:  Φ_min = arccos(1/√3) = 54.74°
    """
    c = np.cos(phi2_rad)
    return np.arccos(c / np.sqrt(1.0 + c**2))


# ── Correct FZ sampling ───────────────────────────────────────────────────────

def sample_cubic_fz(n_grains: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n_grains orientations EXACTLY uniformly from the cubic FZ.

    Returns
    -------
    orientations : (n_grains, 3)  degrees  [phi1, Phi, phi2]
    All values satisfy the paper's FZ definition.
    """
    phi1_out = []
    Phi_out  = []
    phi2_out = []

    while len(phi1_out) < n_grains:
        # Oversample to cover rejection losses (~9%)
        batch = max((n_grains - len(phi1_out)) * 3, 100)

        # ── Step 1: φ₁ uniform in [0, 2π) ───────────────────────────────────
        phi1 = rng.uniform(0.0, 2*np.pi, batch)

        # ── Step 2: φ₂ weighted by cos(φ₂)/√(1+cos²(φ₂)) ───────────────────
        # Sample from Uniform(0, π/4) and apply acceptance criterion.
        # max weight is at φ₂=0: cos(0)/√(1+1) = 1/√2
        phi2 = rng.uniform(0.0, np.pi/4, batch)
        c2   = np.cos(phi2)
        weight_phi2 = c2 / np.sqrt(1.0 + c2**2)
        max_weight  = 1.0 / np.sqrt(2.0)
        accept      = rng.uniform(0.0, 1.0, batch) < weight_phi2 / max_weight
        phi1 = phi1[accept]
        phi2 = phi2[accept]
        n_acc = len(phi1)

        if n_acc == 0:
            continue

        # ── Step 3: Φ from sin-weighted distribution in [Φ_min(φ₂), π/2] ────
        # Inverse CDF: Φ = arccos( cos(Φ_min) × U ),  U ~ Uniform(0,1)
        # Derivation:
        #   PDF(Φ) ∝ sin(Φ)  on [Φ_min, π/2]
        #   CDF(Φ) = (cos(Φ_min) - cos(Φ)) / cos(Φ_min)  [cos(π/2)=0]
        #   Invert: cos(Φ) = cos(Φ_min) × (1 - U)
        #   Let U' = 1 - U ~ Uniform(0,1) → Φ = arccos(cos(Φ_min) × U')
        Phi_min_vals = phi_min(phi2)                     # (n_acc,)
        U   = rng.uniform(0.0, 1.0, n_acc)
        Phi = np.arccos(np.cos(Phi_min_vals) * U)       # (n_acc,)

        phi1_out.extend(phi1.tolist())
        Phi_out.extend(Phi.tolist())
        phi2_out.extend(phi2.tolist())

    # Trim to exact size
    phi1 = np.array(phi1_out[:n_grains])
    Phi  = np.array(Phi_out[:n_grains])
    phi2 = np.array(phi2_out[:n_grains])

    # Convert to degrees for DAMASK .geom files
    return np.degrees(np.column_stack([phi1, Phi, phi2]))


# ── Verification helper ───────────────────────────────────────────────────────

def in_fz(orientations_deg: np.ndarray) -> np.ndarray:
    """
    Return boolean mask: True if orientation is in the paper's cubic FZ.
    orientations_deg: (..., 3) degrees [phi1, Phi, phi2]
    """
    phi2 = np.radians(orientations_deg[..., 2])
    Phi  = np.radians(orientations_deg[..., 1])
    Phi_min_vals = phi_min(phi2)
    return (
        (orientations_deg[..., 2] >= 0.0) &
        (orientations_deg[..., 2] <= 45.0 + 1e-6) &
        (Phi >= Phi_min_vals - 1e-6) &
        (Phi <= np.pi/2 + 1e-6)
    )


# ── DAMASK geom file helpers ──────────────────────────────────────────────────

def microstructure_block(n_grains: int) -> str:
    lines = []
    for g in range(1, n_grains+1):
        lines.append(f"[Grain{g:02d}]")
        lines.append("crystallite 1")
        lines.append(f"(constituent)\tphase 1\ttexture {g:2d}\tfraction 1.0")
    return '\n'.join(lines)


def texture_block(orientations_deg: np.ndarray) -> str:
    lines = []
    for g, (phi1, Phi, phi2) in enumerate(orientations_deg, 1):
        lines.append(f"[Grain{g:02d}]")
        lines.append(f"(gauss)\tphi1 {phi1:.4f}\tPhi {Phi:.4f}\tphi2 {phi2:.4f}"
                     f"\tscatter 0.0\tfraction 1.0")
    return '\n'.join(lines)


def geometry_block(grain_map: np.ndarray) -> str:
    rows = []
    for row in range(grain_map.shape[0]):
        rows.append(' '.join(f'{grain_map[row,col]:2d}'
                             for col in range(grain_map.shape[1])))
    return '\n'.join(rows)


def voronoi_grain_map(n_grains: int,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Generate a 31×31 grain map that is explicitly periodic.

    Strategy
    --------
    1. Tessellate a 30×30 base grid using periodic Voronoi seeds
       (9-image replication so grain boundaries wrap at 30-cell period).
    2. Construct the 31×31 output by tiling the wrap-around boundary:

         col 30  = col  0   (right  column = left  column)
         row  0  = row 29   (top    row    = bottom row  )

    Layout of the 31×31 output (row 0 = top in DAMASK geom convention):

         row  0  : ← copy of row 29 of the 30×30 base (bottom wraps to top)
         rows 1–30: ← rows 0–29 of the 30×30 base
         col  0  : as computed
         cols 1–30: cols 0–29 of the 30×30 base
         col 30  : ← copy of col 0 (left wraps to right)

    This makes periodicity explicit and visible to DAMASK — the grain
    that touches the right edge is the same grain touching the left edge,
    and the grain at the top is the same as the one at the bottom.

    Returns
    -------
    grain_map : (31, 31) int array, grain IDs 1..n_grains
    """
    BASE = 30   # underlying periodic cell size

    # Periodic Voronoi on the BASE×BASE cell
    seeds   = rng.uniform(0.0, 1.0, (n_grains, 2))
    offsets = np.array([[dx,dy] for dx in [-1,0,1] for dy in [-1,0,1]])
    seeds_p = (seeds[None,:,:] + offsets[:,None,:]).reshape(-1, 2)

    ii = (np.arange(BASE) + 0.5) / BASE
    jj = (np.arange(BASE) + 0.5) / BASE
    gi, gj = np.meshgrid(ii, jj, indexing='ij')
    coords  = np.stack([gi, gj], axis=-1)                  # (30, 30, 2)
    dists   = np.sum((coords[:,:,None,:] -
                      seeds_p[None,None,:,:])**2, axis=-1)  # (30, 30, 9*n)
    base_map = np.argmin(dists, axis=-1) % n_grains + 1    # (30, 30)

    # Build 31×31 with explicit periodicity
    # Rows: [row29_of_base, rows0..29_of_base]   → top row = bottom row
    # Cols: [cols0..29_of_base, col0_of_base]    → right col = left col
    out = np.empty((31, 31), dtype=base_map.dtype)
    out[0,  0:30] = base_map[29, :]          # top    row = bottom row of base
    out[1:, 0:30] = base_map[0:30, :]        # rows 1–30 = base rows 0–29
    out[:, 30]    = out[:, 0]                # right col = left col
    return out


def write_geom(path: str, grain_map: np.ndarray,
               orientations_deg: np.ndarray) -> None:
    """
    Write a DAMASK .geom file.
    grain_map must be (31, 31) constructed by voronoi_grain_map() above.
    The grid header is always 31×31×1.
    """
    n_grains  = len(orientations_deg)
    ms_block  = microstructure_block(n_grains)
    tx_block  = texture_block(orientations_deg)
    geo_block = geometry_block(grain_map)

    header_lines = [
        "geom_fromVoronoiTessellation v2.0.3 -g 31 31 1",
        "grid\ta 31\tb 31\tc 1",
        "size\tx 1.0\ty 1.0\tz 0.032258064516129031",
        "origin\tx 0.0\ty 0.0\tz 0.0",
        "homogenization\t1",
        f"microstructures\t{n_grains}",
        "<microstructure>",
    ]
    header_lines.extend(ms_block.split('\n'))
    header_lines.append("<texture>")
    header_lines.extend(tx_block.split('\n'))
    header_lines.append("<!skip>")
    n_header = len(header_lines)

    with open(path, 'w') as f:
        f.write(f"{n_header}\theader\n")
        f.write('\n'.join(header_lines) + '\n')
        f.write(geo_block + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_micros: int, outdir: str, n_grains: int, seed: int) -> np.ndarray:

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    PHI_MAX_DEG = np.degrees(np.arccos(1.0/np.sqrt(3.0)))   # 54.74°

    print(f"Generating {n_micros} geom files  [paper FZ definition]")
    print(f"  Grid    : 31×31 (30×30 Voronoi base + periodic wrap)")
    print(f"            right col = left col,  top row = bottom row")
    print(f"  FZ      : φ₂ ∈ [0°, 45°]")
    print(f"            Φ  ∈ [Φ_min(φ₂), 90°]")
    print(f"            Φ_min ranges from 45.00° (φ₂=0°) to {PHI_MAX_DEG:.2f}° (φ₂=45°)")
    print(f"  Sampling: exact inverse-CDF, no rejection bias")
    print(f"  Output  : {outdir}/")
    print()

    all_orientations = []

    for i in range(1, n_micros+1):
        orientations = sample_cubic_fz(n_grains, rng)   # (n_grains, 3) degrees
        grain_map    = voronoi_grain_map(n_grains, rng) # (31,31) periodic

        fname = os.path.join(outdir, f"d_{i}.geom")
        write_geom(fname, grain_map, orientations)
        all_orientations.append(orientations)

        if i % 100 == 0 or i == n_micros:
            print(f"  Written {i}/{n_micros}")

    all_ori = np.array(all_orientations)   # (n_micros, n_grains, 3) degrees
    np.save(os.path.join(outdir, "all_orientations.npy"), all_ori)

    # ── Verification ─────────────────────────────────────────────────────────
    Phi_all  = all_ori[:,:,1]
    phi2_all = all_ori[:,:,2]
    phi1_all = all_ori[:,:,0]
    fz_mask  = in_fz(all_ori)

    print(f"\nVerification:")
    print(f"  phi1 : [{phi1_all.min():.2f}°, {phi1_all.max():.2f}°]  "
          f"(expect [0°, 360°))")
    print(f"  Phi  : [{Phi_all.min():.2f}°, {Phi_all.max():.2f}°]  "
          f"(expect [{45:.2f}°, 90.00°])")
    print(f"  phi2 : [{phi2_all.min():.2f}°, {phi2_all.max():.2f}°]  "
          f"(expect [0°, 45°])")
    print(f"  Phi  mean = {Phi_all.mean():.2f}°  "
          f"(uniform in [45°,90°] would give 67.5°; "
          f"sin-weighted gives ~71°)")
    print(f"  phi2 mean = {phi2_all.mean():.2f}°  "
          f"(weighted mean, expect ~19°)")
    print(f"  In FZ: {fz_mask.sum()}/{fz_mask.size}  "
          f"({'✓ all' if fz_mask.all() else '✗ some outside'})")

    print(f"\nDone. {n_micros} geom files written to {outdir}/")
    print()
    print("IMPORTANT: orientations are already in the paper's cubic FZ.")
    print("  Pipeline step 4 (FZ mapping) must use the paper's FZ convention,")
    print("  NOT the minimum-Phi convention. Use pipeline_theta0_lp11_paperfz.py")

    return all_ori


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DAMASK geom files with correct cubic FZ sampling")
    parser.add_argument("--n",      type=int,      default=2000)
    parser.add_argument("--outdir", default="./geom_fz_updated")
    parser.add_argument("--grains", type=int,      default=25)
    parser.add_argument("--seed",   type=int,      default=42)
    args = parser.parse_args()

    main(args.n, args.outdir, args.grains, args.seed)
