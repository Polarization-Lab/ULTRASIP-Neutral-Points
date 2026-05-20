# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:23:43 2026

@author: deleo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import statsmodels.api as sm


# ------------------------ Build matrices ------------------------

# 21 values from 20 to 25
sequence2 = np.linspace(20, 25, 21)

# Repeat each value across 10 columns
matrix2 = np.tile(sequence2[:, None], (1, 10))

print(matrix2)
print(matrix2.shape)

# Make sequence with 0 at 3rd row (index 2)
sequence = np.arange(2, -19, -1)   # 2 → -18 (21 values)

# Build Q matrix
matrix_q = np.tile(sequence[:, None], (1, 20))


# ------------------------ Colormap normalization ------------------------

# Force 0 to be white
norm = TwoSlopeNorm(vmin=sequence.min(), vcenter=0, vmax=sequence.max())


# ------------------------ First image ------------------------

plt.figure()
im = plt.imshow(matrix_q, cmap='RdBu_r', norm=norm)
cbar = plt.colorbar(im)
cbar.set_ticks([sequence.min(), 0, sequence.max()])
cbar.set_ticklabels([f"{sequence.min()}", "0", f"{sequence.max()}"])
plt.axis('off')


# ------------------------ Derived values ------------------------

r_q = np.average(matrix_q, axis=1)  # row averages
vza = matrix2[:, 0]                 # corresponding VZA


# ------------------------ WLS regression ------------------------

# VZA = a + b * r_Q
X = sm.add_constant(r_q)
y = vza

weights = np.ones_like(y)  # equal weights

model = sm.WLS(y, X, weights=weights)
results = model.fit()

intercept = results.params[0]
slope = results.params[1]

# Zero-crossing (r_Q = 0)
vza_zero = intercept

print(results.summary())
print(f"\nZero-crossing VZA (r_Q = 0): {vza_zero:.3f}")


# ------------------------ Scatter + fit ------------------------

plt.figure()
plt.scatter(r_q, vza, label='Data')

# Fit line
r_fit = np.linspace(min(r_q), max(r_q), 100)
vza_fit = intercept + slope * r_fit
plt.plot(r_fit, vza_fit, 'k-', label='WLS fit')

# Zero-crossing line
plt.axhline(vza_zero, linestyle='--', label=f'Zero crossing = {vza_zero:.2f}')

plt.xlabel("r_Q")
plt.ylabel("VZA")
plt.legend()


# ------------------------ Second image with VZA axis ------------------------

plt.figure()
im = plt.imshow(
    matrix_q,
    cmap='RdBu_r',
    norm=norm,
    aspect='auto',
    extent=[0, 21, vza.max(), vza.min()]
)

cbar = plt.colorbar(im)
cbar.set_ticks([sequence.min(), 0, sequence.max()])
cbar.set_ticklabels([f"{sequence.min()}", "0", f"{sequence.max()}"])

plt.ylabel("VZA")
plt.xlabel("pixels")

plt.show()

# Use your matrix (matrix_q or whatever you're working with)
img_x, img_y = matrix_q.shape   # <-- correct unpacking

HFOV = VFOV = 0.002
Pan = 22
Tilt = 60
img_x = 2848
img_y = 2848
x_center = img_x / 4
y_center = img_y / 4

delta_zen = 0

# Create grid
x = np.arange(img_x)
y = np.arange(img_y)
xx, yy = np.meshgrid(x, y, indexing='ij')

# Pixel angular offsets
x_dev = (x_center - xx) * HFOV   # vertical direction
y_dev = (y_center - yy) * VFOV   # horizontal direction

# Compute view angles
view_az = Pan - y_dev
view_zen = Tilt - x_dev