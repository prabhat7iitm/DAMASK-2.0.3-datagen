#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 00:38:52 2025

@author: prabhat
"""

import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Base path where geom_file folder will be created
base_path = "/home/Crystal_Plasticity/EPP/Composite/Vol_frac/20/"
output_path = os.path.join(base_path, "geom_file")  # Unified output folder
os.makedirs(output_path, exist_ok=True)

# Load the input data
with open(os.path.join(base_path, 'vol_20.npy'), 'rb') as f:
    input_ms = np.load(f).reshape([-1, 100, 100])  # Shape: (-1, 200, 200)


def generate_periodic_voronoi_file(grid, file_path):
    """Generate and save a periodic Voronoi microstructure file."""
    num_grains = 2
    grid_size = grid.shape[0]

    with open(file_path, "w") as f:
        f.write("19\theader\n")
        f.write(f"geom_fromVoronoiTessellation v2.0.3 -g {grid_size} {grid_size} 1\n")
        f.write(f"grid\ta {grid_size}\tb {grid_size}\tc 1\n")
        f.write("size\tx 1.0\ty 1.0\tz 0.01\n")
        f.write("origin\tx 0.0\ty 0.0\tz 0.0\n")
        f.write("homogenization\t1\n")
        f.write("microstructures\t2\n")
        f.write("<microstructure>\n")

        for i in range(1, num_grains + 1):
            f.write(f"[Grain{i:03d}]\n")
            f.write("crystallite 1\n")
            f.write(f"(constituent)\tphase {i % 2 + 1}\ttexture {i % 2 + 1}\tfraction 1.0\n")

        f.write("<texture>\n")
        for i in range(1, num_grains + 1):
            phi1, Phi, phi2 = 0, 0, 0
            f.write(f"[Grain{i:03d}]\n")
            f.write(f"(gauss)\tphi1 {phi1:.4f}\tPhi {Phi:.4f}\tphi2 {phi2:.4f}\tscatter 0.0\tfraction 1.0\n")

        f.write("<!skip>\n")
        for row in grid:
            f.write(" ".join(map(str, row + 1)) + "\n")


def save_microstructure_plot(grid, file_path):
    """Save the microstructure plot as an image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='gray', origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

#%%
# Loop over all samples and save to the same folder
for i in tqdm(range(1, input_ms.shape[0] + 1), desc="Generating files", unit="file"):
    grid = input_ms[i - 1]

    geom_filename = os.path.join(output_path, f"d_{i}.geom")
    png_filename = os.path.join(output_path, f"d_{i}.png")

    generate_periodic_voronoi_file(grid, geom_filename)
    save_microstructure_plot(grid, png_filename)

print("All .geom and .png files have been saved in 'geom_file' successfully!")
