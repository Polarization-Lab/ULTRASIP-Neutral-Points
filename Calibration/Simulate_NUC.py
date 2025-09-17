# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 10:42:52 2025

@author: C.M.DeLeon
"""
#NUC Simulation 
#Import Libraries 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Image size
Ni, Nj = 1024, 1024

Rij = np.zeros((Ni, Nj))
Bij = np.zeros((Ni, Nj))

# Create image
x = np.linspace(0, Nj-1, Nj)
y = np.linspace(0, Ni-1, Ni)
X, Y = np.meshgrid(x, y)

cx, cy = Nj // 2, Ni // 2  # center of the Gaussian
sigma = 500.0

# Create Gaussian image
Pij = 23 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
Pij = Pij + np.random.normal(0, 1, Pij.shape)  # Add Gaussian noise first

# Create synthetic exposures
exposure_times = np.array([200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900])
# Generate image stack by scaling base image
stacked = np.array([Pij * exp for exp in exposure_times])

for i in range(0,Ni):
    for j in range(0,Nj):
        pixel_values = stacked[:, i, j]
        slope, intercept, *_ = linregress(exposure_times, pixel_values)
        Rij[i, j] = slope
        Bij[i, j] = intercept

# Get global slope/intercept
R_avg = np.mean(Rij)
B_avg = np.mean(Bij)

# Apply NUC correction
Cij = (R_avg / Rij) * (Pij - Bij) + B_avg


plt.figure()
plt.imshow(Pij, interpolation='None', cmap='gray')
plt.colorbar()
plt.title(r'P$_{ij}$ with Gaussian Noise')
plt.show()

plt.figure()
plt.scatter(exposure_times, pixel_values, color='blue')
plt.xlabel('Exposure Value')
plt.ylabel(f'Pixel Intensity at ({i},{j})')
plt.title(f'Pixel Response: ({i},{j})')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting
plt.figure()
plt.imshow(Rij, interpolation='None', cmap='gray')
plt.colorbar()
plt.title('R$_{ij}$')

plt.figure()
plt.imshow(Bij, interpolation='None', cmap='gray')
plt.colorbar()
plt.title('B$_{ij}$')

plt.figure()
plt.imshow(Cij, interpolation='None', cmap='gray',vmin=58,vmax=60)
plt.colorbar()
plt.title('Corrected Image C$_{ij}$')

plt.figure()
plt.hist(Cij.flatten())




