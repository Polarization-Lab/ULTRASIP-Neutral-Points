# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 15:06:27 2025
@author: deleo

Create NUC

"""

#Import libraries 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import cmocean.cm as cmo
import glob
import h5py
import os

# --- Config ---
cal_type = 'NUC'  # 'NUC' or 'Malus'
cal_path = 'E:/Calibration/Data'
cal_files = glob.glob(f'{cal_path}/{cal_type}*.h5')

idx = 6 #len(cal_files) - 3  # choose file index #8,7,6,5
Ni, Nj = 2848, 2848       # image size

with h5py.File(cal_files[idx], 'r+') as f:

    # --- Load exposures once ---
    exp_times = f['P_0 Measurements/Exposure Times'][:]

    # --- Loop over angles ---
    angles = [0, 45, 90, 135]
    Rij, Bij = {}, {}

    for ang in angles:
        uvimgs = f[f'P_{ang} Measurements/UV Raw Images'][:]
        meas = uvimgs.reshape(len(exp_times), Ni, Nj)

        # Linear regression via least squares (vectorized)
        A = np.vstack([exp_times, np.ones_like(exp_times)]).T  # shape (n,2)
        coeffs, _, _, _ = np.linalg.lstsq(A, meas.reshape(len(exp_times), -1), rcond=None)
        slope = coeffs[0].reshape(Ni, Nj)
        intercept = coeffs[1].reshape(Ni, Nj)

        Rij[ang], Bij[ang] = slope, intercept
        
fig=plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.title("R0ij")
plt.imshow(Rij[0], cmap='gray',interpolation = 'None')
plt.colorbar(shrink=0.5)

plt.subplot(1,4,2)
plt.title("R90ij")
plt.imshow(Rij[90], cmap='gray',interpolation ='None')
plt.colorbar(shrink=0.5)

plt.subplot(1,4,3)
plt.title("R45ij")
plt.imshow(Rij[45], cmap='gray',interpolation ='None')
plt.colorbar(shrink=0.5)


plt.subplot(1,4,4)
plt.title("R135ij")
plt.imshow(Rij[135], cmap='gray',interpolation ='None')
plt.colorbar(shrink=0.5)

fig.suptitle("Rij Images", fontsize=18, y=0.8)
plt.tight_layout()
plt.show()

fig=plt.figure(figsize=(15,5))
plt.subplot(1,4,1)
plt.title("B0ij")
plt.imshow(Bij[0], cmap='gray',interpolation = 'None',vmin=-10)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,2)
plt.title("B90ij")
plt.imshow(Bij[90], cmap='gray',interpolation ='None',vmin=-10)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,3)
plt.title("B45ij")
plt.imshow(Bij[45], cmap='gray',interpolation ='None',vmin=-10)
plt.colorbar(shrink=0.5)


plt.subplot(1,4,4)
plt.title("B135ij")
plt.imshow(Bij[135], cmap='gray',interpolation ='None',vmin=-10)
plt.colorbar(shrink=0.5)

fig.suptitle("Bij Images", fontsize=18, y=0.8)
plt.tight_layout()
plt.show()
        
        

np.savez('E:/NUC_0812_1643.npz', arr1=Rij, arr2=Bij)
