import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------
base_path = "/home/pCrystal_plasticity/Morphology/2D/Goss/Y_elongated/dataset_2000_cases/"
output_path = os.path.join(base_path, "geom_file")
os.makedirs(output_path, exist_ok=True)

grain_path = os.path.join(base_path, "grains")
euler_path = os.path.join(base_path, "eulers")

# --------------------------------------------------
# DAMASK .geom writer
# --------------------------------------------------
def generate_geom_from_grains_and_euler(grains, euler_map, file_path):

    ny, nx = grains.shape
    grain_ids = np.unique(grains)
    n_grains = len(grain_ids)

    with open(file_path, "w") as f:

        # ---------------- Header ----------------
        f.write("59\theader\n")
        f.write(f"geom_fromVoronoiTessellation v2.0.3 -g {nx} {ny} 1\n")
        f.write(f"grid\ta {nx}\tb {ny}\tc 1\n")
        f.write("size\tx 1.0\ty 1.0\tz 0.01\n")
        f.write("origin\tx 0.0\ty 0.0\tz 0.0\n")
        f.write("homogenization\t1\n")
        f.write(f"microstructures\t{n_grains}\n")

        # ---------------- Microstructure ----------------
        f.write("<microstructure>\n")
        for i, gid in enumerate(grain_ids, start=1):
            f.write(f"[Grain{i:02d}]\n")
            f.write("crystallite 1\n")
            f.write(f"(constituent)\tphase 1\ttexture {i}\tfraction 1.0\n")

        # ---------------- Texture ----------------
        f.write("<texture>\n")
        for i, gid in enumerate(grain_ids, start=1):

            # Extract one Euler angle per grain
            mask = grains == gid
            phi1, Phi, phi2 = np.unique(euler_map[mask], axis=0)[0]

            f.write(f"[Grain{i:02d}]\n")
            f.write(
                f"(gauss)\tphi1 {phi1:.4f}\t"
                f"Phi {Phi:.4f}\t"
                f"phi2 {phi2:.4f}\t"
                f"scatter 0.0\tfraction 1.0\n"
            )

        # ---------------- Geometry ----------------
        f.write("<!skip>\n")
        for row in grains:
            f.write(" ".join(map(str, row + 1)) + "\n")


# --------------------------------------------------
# Plot helper
# --------------------------------------------------
def save_microstructure_plot(grid, file_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="tab20", origin="lower")
    plt.axis("off")
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


# --------------------------------------------------
# Main loop
# --------------------------------------------------
n_cases = 2000

for i in tqdm(range(n_cases), desc="Generating DAMASK geom files"):

    grains = np.load(os.path.join(grain_path, f"grains_{(i+1)}.npy"))
    euler_map = np.load(os.path.join(euler_path, f"euler_{(i+1)}.npy"))

    geom_file = os.path.join(output_path, f"d_{(i+1)}.geom")
    png_file  = os.path.join(output_path, f"d_{(i+1)}.png")

    generate_geom_from_grains_and_euler(grains, euler_map, geom_file)
    save_microstructure_plot(grains, png_file)

print("All DAMASK .geom and .png files created successfully!")
