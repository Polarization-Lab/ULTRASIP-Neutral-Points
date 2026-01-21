# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 14:41:36 2025

@author: C.M.DeLeon

NUC Data Analysis - make sure the flat fields are good

"""
#Import libraries 
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import datetime
import cmocean.cm as cmo
import numpy as np
import glob
import h5py
import os

data = np.load('D:/NUC_0813.npz', allow_pickle=True)

Rij = data['arr1'].item()   # convert array-object → Python dict
Bij = data['arr2'].item()

#Correct image
def correct_img(Pij,Rij,Bij):
    R_avg = np.mean(Rij)
    B_avg = np.mean(Bij)
    Cij = (R_avg / Rij) * (Pij - Bij) + B_avg  
    return Cij
# P0 = 1500*np.ones((2848,2848))
# P45 = 0.5*1500*np.ones((2848,2848))
# P135 = 0.5*1500*np.ones((2848,2848))
# P90 = 0.01*1500*np.ones((2848,2848))

cal_type = 'NUC'  # 'NUC' or 'Malus'
cal_path = 'D:/Calibration/Data'
cal_files = glob.glob(f'{cal_path}/{cal_type}*.h5')

idx = 8 #len(cal_files) - 3  # choose file index #8,7,6,5
Ni, Nj = 2848, 2848       # image size

# Open HDF5 calibration file
with h5py.File(cal_files[idx], 'r+') as f:

    angles = [0, 45, 90, 135]

    # dictionary to store the output images
    P_images = {}
    P_images_corrected = {}

    exp_times = f['P_0 Measurements/Exposure Times'][:]
    Nexp = len(exp_times)
    run=Nexp-1

    for ang in angles:
        # Use temporary variables for per-angle Rij/Bij
        Rij_ang = Rij[ang]
        Bij_ang = Bij[ang]

        # Load raw UV images
        uvimgs = f[f'P_{ang} Measurements/UV Raw Images'][:]
        meas = uvimgs.reshape(Nexp, Ni, Nj)

        # Correct the longest-exposure image
        Cij = correct_img(meas[-1], Rij_ang, Bij_ang)

        # Store in dict
        #P_images[ang] = meas[-1]
        P_images[ang] = Cij   # shape = (Ni, Nj)

# Now extract them in correct Stokes order
P0   = P_images[0]
P90  = P_images[90]
P45  = P_images[45]
P135 = P_images[135]

# Stack into Stokes order (I,Q,U)
P = np.stack([P0, P90, P45, P135], axis=-1)

cmin=2400
cmax=3000
fig=plt.figure(figsize=(17,5))

plt.subplot(1,4,1)
plt.title("P0")
plt.imshow(P0, cmap='gray',interpolation = 'None',vmin=cmin,vmax=cmax)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,2)
plt.title("P90")
plt.imshow(P90, cmap='gray',interpolation ='None',vmin=cmin,vmax=cmax)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,3)
plt.title("P45")
plt.imshow(P45, cmap='gray',interpolation ='None',vmin=cmin,vmax=cmax)
plt.colorbar(shrink=0.5)


plt.subplot(1,4,4)
plt.title("P135")
plt.imshow(P135, cmap='gray',interpolation ='None',vmin=cmin,vmax=cmax)
plt.colorbar(shrink=0.5)

fig.suptitle(f"Cij Images — Exposure Time = {exp_times[run]:.6f} us", fontsize=18, y=0.9)
plt.tight_layout()
plt.show()

# Load pixel-wise W matrix
W = np.load('D:/ULTRASIP_AvgWmatrix_15.npy')       # shape = (H, W, 4, 3)

# Wi = 0.5 * np.array([
#     [1,  1,  0],
#     [1, -1,  0],
#     [1,  0,  1],
#     [1,  0, -1]
# ])  # shape (4, 3)
# # #Broadcast W to a full 2848 x 2848 grid
# W = np.broadcast_to(W, (2848, 2848, 4, 3)).copy()


H, Wd = P.shape[0], P.shape[1]
# # --- Pixelwise pseudoinverse via SVD ---
# Compute SVD of each 4x3 matrix
U, S, Vt = np.linalg.svd(W, full_matrices=False)   # shapes:
# U:  (H, W, 4, 3)
# S:  (H, W, 3)
# Vt: (H, W, 3, 3)

# Invert singular values 
S_inv = np.zeros_like(S)
S_inv = np.where(S > 1e-12, 1/S, 0)                # avoids division by zero

# Reconstruct pseudoinverse:  pinv(W) = V * S^-1 * U^T
pinvW = np.matmul(Vt.transpose(0,1,3,2) * S_inv[...,None,:], 
                  U.transpose(0,1,3,2))            # shape = (H,W,3,4)


# Apply per-pixel calibration
# Stokes = pinv(W) @ P_per_pixel
Stokes = np.matmul(pinvW, P[...,None])[...,0]       # shape = (H, W, 3)


# Stokes is assumed to be shape (H, W, 3)
I = Stokes[:,:,0]
Q = Stokes[:,:,1]
U = Stokes[:,:,2]

avgQ = np.flip(np.average(Q/I,axis=1)) #row avg
avgU = np.average(U/I,axis=0)#col avg


dolp = (np.sqrt((Q**2)+(U**2))/I)*100
dolp_mean = np.average(dolp)
dolp_std = np.std(dolp)
dolp_median = np.median(dolp)

dolp_avg = np.sqrt((avgQ**2)+(avgU**2))*100
dolpavg_mean = np.average(dolp_avg)
dolpavg_std = np.std(dolp_avg)
dolpavg_median = np.median(dolp_avg)
    
aolp = 0.5*np.arctan2(U,Q)
aolp = np.mod(np.degrees(aolp),180)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title(" I")
plt.imshow(I, cmap='gray',interpolation = 'None')
plt.colorbar(shrink=0.75)

plt.subplot(1,3,2)
plt.title(" Q/I")
plt.imshow(Q/I, cmap=cmo.curl,interpolation ='None',vmin=-0.1,vmax=0.1)
plt.colorbar(shrink=0.75)

plt.subplot(1,3,3)
plt.title(" U/I")
plt.imshow(U/I, cmap=cmo.curl,interpolation ='None',vmin=-0.1,vmax=0.1)
plt.colorbar(shrink=0.75)

plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(dolp, cmap='hot', interpolation = 'None',vmin=0,vmax=8)
plt.title('DoLP [%]',fontsize=20)
plt.colorbar()

plt.figure()
plt.imshow(aolp, cmap=cmo.phase, interpolation = 'None',vmin=0,vmax=180)
plt.title('AoLP  [deg]',fontsize=20)
plt.colorbar()

plt.figure()
plt.hist(aolp.flatten())
plt.title('AoLP [deg]')
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(dolp.flatten(),range=(0, 8))
ax.set_title('DoLP  [%]')
# Add text box to the right of plot
textstr = f"Mean = {dolp_mean:.4f}%\nStd = {dolp_std:.4f}%\nMed = {dolp_median:.4f}%"
ax.text(0.5, 0.5, textstr, transform=ax.transAxes, fontsize=14,
verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
            
# Scatter plot
ax.scatter( range(len(dolp_avg)),dolp_avg, color='green')
ax.set_ylabel(r'$DoLP_{rc} [\%]$', fontsize=15)
ax.set_xlabel('Pixel Index', fontsize=15)
ax.set_title(r'DoLP from $\bar{c}_{U},\bar{r}_{Q}$', fontsize=16)
            
# Add text box to the right of plot
textstr = f"Mean = {dolpavg_mean:.4f}%\nStd = {dolpavg_std:.4f}%\nMed = {dolpavg_median:.4f}%"
ax.text(1.05, 0.5, textstr, transform=ax.transAxes, fontsize=14,
verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.show()

#np.save('D:/ULTRASIP_Avg_Wmatrix.npy', W)
