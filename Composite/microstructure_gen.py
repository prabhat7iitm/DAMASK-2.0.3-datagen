#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 23:53:18 2023

@author: prabhat
"""

import numpy as np
from sklearn.pipeline import Pipeline
import dask.array as da
import random
from random import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pymks import (
    generate_delta,
    generate_multiphase,
    solve_fe,
    plot_microstructures,
    coeff_to_real
)
import warnings
warnings.filterwarnings('ignore')


def shuffle(data):
    tmp = np.array(data)
    np.random.shuffle(tmp)
    return da.from_array(tmp, chunks=data.chunks)


da.random.seed(10)
np.random.seed(10)

tmp = [
    generate_multiphase(shape=(500,100, 100), grain_size=x, volume_fraction=(0.2, 0.8), chunks=250, percent_variance=0.001)
    for x in [(20, 20)]
]
x_data = shuffle(da.concatenate(tmp))

#%%
int_ = random.randint(0, 199)
print(int_)
import matplotlib.pyplot as plt

# Assume x_data[int_, :, :] contains your 2D microstructure grid
microstructure = x_data[int_, :, :]  # Replace int_ with your index

plt.figure(figsize=(6, 6))
im = plt.imshow(microstructure, cmap='Greys', origin='lower', vmin=0, vmax=1)
plt.title("Microstructure")
plt.axis("off")  # Hide axis ticks

# Add colorbar with range 0 to 1
cbar = plt.colorbar(im, shrink=0.8)
cbar.set_label("Grain ID (normalized)")  # Optional label
cbar.set_ticks([0, 0.5, 1])  # Optional: fixed tick marks

# Save the figure
plt.savefig("microstructure_plot.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()



X = x_data[int_]
X = np.array(X)
p = sum(sum(X==1))
# #%%
path = "/home/karmakar/Crystal_Plasticity/EPP/Composite/Vol_frac/20/"
np.save(path + "vol_20.npy",x_data)
volfrac_ = p/(200*200)
print(volfrac_)
#%%

