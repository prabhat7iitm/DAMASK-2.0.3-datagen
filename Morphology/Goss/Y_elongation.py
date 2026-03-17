import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from tqdm import tqdm

# ==================================================
# 1. Global parameters
# ==================================================
nx, ny = 100, 100
n_grains = 10
n_cases = 2001

# Ideal Goss orientation (Bunge, degrees)
goss_euler = np.array([0.0, 45.0, 0.0])
misorientation_std = 5.0  # degrees

# Grain morphology
sigma_x, sigma_y = 2, 5   # elongation along Y

# ==================================================
# 2. Output paths
# ==================================================
data_path = "dataset_2000_cases"

grain_dir = os.path.join(data_path, "grains")
euler_dir = os.path.join(data_path, "eulers")
img_dir   = os.path.join(data_path, "images")

os.makedirs(grain_dir, exist_ok=True)
os.makedirs(euler_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# ==================================================
# 3. Case loop
# ==================================================
for case_id in tqdm(range(n_cases), desc="Generating cases"):

    np.random.seed(case_id)  # reproducibility

    # --------------------------------------------------
    # 3.1 Generate interior grain map
    # --------------------------------------------------
    nx_i, ny_i = nx - 1, ny - 1
    grain_labels = np.random.randint(0, n_grains, size=(ny_i, nx_i))

    one_hot = np.zeros((n_grains, ny_i, nx_i), dtype=np.float32)
    for g in range(n_grains):
        one_hot[g] = (grain_labels == g).astype(np.float32)

    # --------------------------------------------------
    # 3.2 Anisotropic Gaussian filtering
    # --------------------------------------------------
    smoothed = np.array([
        gaussian_filter(one_hot[g], sigma=[sigma_y, sigma_x], mode="wrap")
        for g in range(n_grains)
    ])

    interior = np.argmax(smoothed, axis=0)

    # --------------------------------------------------
    # 3.3 Enforce periodicity
    # --------------------------------------------------
    grains_2D = np.zeros((ny, nx), dtype=np.int32)
    grains_2D[:-1, :-1] = interior
    grains_2D[-1, :-1]  = grains_2D[0, :-1]
    grains_2D[:-1, -1]  = grains_2D[:-1, 0]
    grains_2D[-1, -1]   = grains_2D[0, 0]

    # --------------------------------------------------
    # 3.4 Assign grain-wise Goss texture
    # --------------------------------------------------
    euler_map = np.zeros((ny, nx, 3), dtype=np.float32)

    for g in range(n_grains):
        delta = np.random.normal(0.0, misorientation_std, size=3)
        euler = goss_euler + delta

        euler[0] %= 360.0
        euler[1] = np.clip(euler[1], 0.0, 180.0)
        euler[2] %= 360.0

        euler_map[grains_2D == g, 0] = euler[0]
        euler_map[grains_2D == g, 1] = euler[1]
        euler_map[grains_2D == g, 2] = euler[2]

    # --------------------------------------------------
    # 3.5 Save arrays
    # --------------------------------------------------
    np.save(
        os.path.join(grain_dir, f"grains_{case_id}.npy"),
        grains_2D
    )

    np.save(
        os.path.join(euler_dir, f"euler_{case_id}.npy"),
        euler_map
    )

    # --------------------------------------------------
    # 3.6 Save quick visualization (optional)
    # --------------------------------------------------
    plt.figure(figsize=(4, 4))
    plt.imshow(grains_2D, cmap="tab20", origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(img_dir, f"microstructure_{case_id}.png"),
        dpi=150
    )
    plt.close()

print(f"\nDone.")
print(f"Grains saved in : {grain_dir}")
print(f"Euler angles in: {euler_dir}")
print(f"Images saved in: {img_dir}")
