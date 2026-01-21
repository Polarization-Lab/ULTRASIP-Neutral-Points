# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:13:01 2025

@author: ULTRASIP_1
"""
#Import libraries 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import cmocean.cm as cmo
import glob
import h5py
import os
# Define W-matrix of ULTRASIP (rows = analyzer vectors P0, P90, P45, P135)
W = 0.5 * np.array([[1, 1, 0],[1, -1, 0],[1, 0, 1],[1, 0, -1]])
#W-matrix cal 
Stokes_ideal = np.array([[1,1,1,1],[1,-1,0,0],[0,0,1,-1]])
data = np.load('D:/NUC_0813.npz', allow_pickle=True)

Rij = data['arr1'].item()   # convert array-object → Python dict
Bij = data['arr2'].item()


#Correct image
def correct_img(Pij,Rij,Bij):
    R_avg = np.mean(Rij)
    B_avg = np.mean(Bij)
    Cij = (R_avg / Rij) * (Pij - Bij) + B_avg
    
    return Cij

Wavg = np.load('D:/ULTRASIP_Avg_Wmatrix.npy')       # shape = (H, W, 4, 3)
W = np.load('D:/ULTRASIP_Wmatrix_15.npy')      
#kappa= kappa[1644:2000,1644:2000] 
W = W[1000:2500,1000:2500]


# cal_path = 'D:/Calibration/Data'
# generator_0_file = glob.glob(f'{cal_path}/Malus*1118_16_02_07*.h5')
# g0 = h5py.File(generator_0_file[0],'r+')
# #Horizontal Measurements generator, analyzer
# runs0 = g0["Measurement_Metadata"].attrs['Runs for each angle']
# P_00 = np.mean(g0["P_0 Measurements/UV Raw Images"][:].reshape(runs0,2848,2848),axis=0)
# P_090 = np.mean(g0["P_90 Measurements/UV Raw Images"][:].reshape(runs0,2848,2848),axis=0)
# P_045 = np.mean(g0["P_45 Measurements/UV Raw Images"][:].reshape(runs0,2848,2848),axis=0)
# P_0135 = np.mean(g0["P_135 Measurements/UV Raw Images"][:].reshape(runs0,2848,2848),axis=0)

# cal_path = 'D:/Calibration/Data'
# generator_0_file = glob.glob(f'{cal_path}/NUC_20250925_15_29_46.h5')
# g0 = h5py.File(generator_0_file[0],'r+')
# exp_times = g0['P_0 Measurements/Exposure Times'][:]
# Nexp = len(exp_times)
# P_00 =   g0["P_0 Measurements/UV Raw Images"][:].reshape(Nexp,2848,2848)
# P_00 = P_00[Nexp-1,:,:]
# P_090 =  g0["P_90 Measurements/UV Raw Images"][:].reshape(Nexp,2848,2848)
# P_090 = P_090[Nexp-1,:,:]
# P_045 =  g0["P_45 Measurements/UV Raw Images"][:].reshape(Nexp,2848,2848)
# P_045 = P_045[Nexp-1,:,:]
# P_0135 = g0["P_135 Measurements/UV Raw Images"][:].reshape(Nexp,2848,2848)
# P_0135 = P_0135[Nexp-1,:,:]


# #Corrected Horizontal
# P0 = correct_img(P_00,Rij[0],Bij[0])
# P90 = correct_img(P_090,Rij[90],Bij[90])
# P45 = correct_img(P_045,Rij[45],Bij[45])
# P135 = correct_img(P_0135,Rij[135],Bij[135])

#Simulate 0deg fluxes 
P90 = 0.5*1500*np.ones((2848,2848))
P135 = 1500*np.ones((2848,2848))
P45 = 0.000001*1500*np.ones((2848,2848))
P0 = 0.5*1500*np.ones((2848,2848))

fig=plt.figure(figsize=(17,5))

cmin = 200
cmax= 1800
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

#fig.suptitle("NUC Corrected Integrating Sphere Measured w/ 0deg Generator Fluxes", fontsize=18, y=0.9)
fig.suptitle("Unpolarized Input", fontsize=18, y=0.9)

plt.tight_layout()
plt.show()

P = np.stack([P0, P90, P45, P135], axis=-1)

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
    
aolp = np.degrees(0.5*np.arctan2(U,Q))
aolp_mean = np.average(aolp)
aolp_std = np.std(aolp)
aolp_median = np.median(aolp)

aolp = np.mod(aolp,180)


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title(" I")
plt.imshow(I, cmap='gray',interpolation = 'None')
plt.colorbar(shrink=0.75)

plt.subplot(1,3,2)
plt.title(" Q/I")
plt.imshow(Q/I, cmap=cmo.curl,interpolation ='None',vmin=-1,vmax=1)
plt.colorbar(shrink=0.75)

plt.subplot(1,3,3)
plt.title(" U/I")
plt.imshow(U/I, cmap=cmo.curl,interpolation ='None',vmin=-1,vmax=1)
plt.colorbar(shrink=0.75)

plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(dolp, cmap='hot', interpolation = 'None',vmin=80,vmax=100)
plt.title('DoLP [%]',fontsize=20)
plt.colorbar()

plt.figure()
plt.imshow(aolp, cmap=cmo.phase, interpolation = 'None',vmin=0,vmax=180)
plt.title('AoLP  [deg]',fontsize=20)
plt.colorbar()


# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(aolp.flatten())
# ax.set_title('AoLP  [$\circ$]',fontsize=16)
# ax.tick_params(axis='both', labelsize=14)
# # Add text box to the right of plot
# textstr = f"Mean = {aolp_mean:.4f}$\circ$\nStd = {aolp_std:.4f}$\circ$\nMed = {aolp_median:.4f}$\circ$"
# ax.text(0.5, 0.5, textstr, transform=ax.transAxes, fontsize=14,
# verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
# plt.show()

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(aolp.flatten(),range=(0,180), bins = 100)
# ax.set_title('AoLP  [$\circ$]',fontsize=16)
# ax.tick_params(axis='both', labelsize=14)
# ax.xaxis.set_major_locator(MultipleLocator(15))   # labeled ticks every 20°
# ax.xaxis.set_minor_locator(MultipleLocator(2))    # small ticks every 5°
# plt.show()

fig, ax = plt.subplots(figsize=(8, 6))

counts, bins, patches = ax.hist(
    aolp.flatten(),
    range=(0, 180),
    bins=100
)

ax.set_title('AoLP  [$\\circ$]', fontsize=16)
ax.tick_params(axis='both', labelsize=14)

ax.xaxis.set_major_locator(MultipleLocator(15))
ax.xaxis.set_minor_locator(MultipleLocator(2))

# Label AoLP bin center (degrees) on top of each bar
for left_edge, count, patch in zip(bins[:-1], counts, patches):
    if count > 0:
        bin_center = left_edge + patch.get_width() / 2
        ax.text(
            bin_center,
            count,
            f'{bin_center:.1f}°',
            ha='center', va='bottom',
            fontsize=14,
            rotation=0
        )

plt.show()



fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(dolp.flatten(),range=(80, 120),bins=100)
ax.set_title('DoLP  [%]',fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.xaxis.set_major_locator(MultipleLocator(10))   
ax.xaxis.set_minor_locator(MultipleLocator(2))   
# Add text box to the right of plot
textstr = f"Mean = {dolp_mean:.4f}%\nStd = {dolp_std:.4f}%\nMed = {dolp_median:.4f}%"
ax.text(0.05, 0.8, textstr, transform=ax.transAxes, fontsize=14,
verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
            
# Scatter plot
ax.scatter( range(len(dolp_avg)),dolp_avg, color='green')
ax.set_ylabel(r'$DoLP_{rc} [\%]$', fontsize=15)
ax.set_xlabel('Pixel Index', fontsize=15)
ax.set_title(r'DoLP from $\bar{c}_{U},\bar{r}_{Q}$', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
# Add text box to the right of plot
textstr = f"Mean = {dolpavg_mean:.4f}%\nStd = {dolpavg_std:.4f}%\nMed = {dolpavg_median:.4f}%"
ax.text(1.05, 0.5, textstr, transform=ax.transAxes, fontsize=14,
verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.show()
