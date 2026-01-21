# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 10:58:08 2026

@author: ULTRASIP_1
"""

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

def pseudoinverse(matrix):
    
    # # --- Pixelwise pseudoinverse via SVD ---
    # Compute SVD of each 4x3 matrix
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)   # shapes:
    # U:  (H, W, 4, 3)
    # S:  (H, W, 3)
    # Vt: (H, W, 3, 3)

    # Invert singular values 
    S_inv = np.zeros_like(S)
    S_inv = np.where(S > 1e-12, 1/S, 0)                # avoids division by zero

    # Reconstruct pseudoinverse:  pinv(W) = V * S^-1 * U^T
    pseudoinv = np.matmul(Vt.transpose(0,1,3,2) * S_inv[...,None,:], 
                      U.transpose(0,1,3,2))            # shape = (H,W,3,4)
    return pseudoinv

#Correct image
def correct_img(Pij,Rij,Bij):
    R_avg = np.mean(Rij)
    B_avg = np.mean(Bij)
    Cij = (R_avg / Rij) * (Pij - Bij) + B_avg
    
    return Cij

#NUC FIles
data = np.load('D:/NUC_0813.npz', allow_pickle=True)
Rij = data['arr1'].item()   # convert array-object → Python dict
Bij = data['arr2'].item()

#Define Ideal W-matrix and Stokes Parameters
# Define W-matrix of ULTRASIP (rows = analyzer vectors P0, P90, P45, P135)
W_ideal = 0.5 * np.array([[1, 1, 0],[1, -1, 0],[1, 0, 1],[1, 0, -1]])
W_ideal = np.broadcast_to(W_ideal, (2848, 2848, 4, 3))
pinvW_ideal = pseudoinverse(W_ideal)

#Load ULTRASIP W-matrix 
W_ULTRASIP = np.load('D:/ULTRASIP_AvgWmatrix_15.npy')
pinvW_ULTRASIP = pseudoinverse(W_ULTRASIP)

#Define Ideal Stokes
Stokes_ideal = np.array([[1,1,1,1,1],[1,-1,0,0,0],[0,0,1,-1,0]])

# #Test on Simulated Data 
# title = 'Simulated Horizontal'
# P = W_ULTRASIP@Stokes_ideal[:,0]
# P0 = P[:,:,0]
# P90 = P[:,:,1]
# P45 = P[:,:,2]
# P135 = P[:,:,3]

# #Test on Simulated Data 
# title = 'Simulated Vertical'
# P = W_ULTRASIP@Stokes_ideal[:,1]
# P0 = P[:,:,0]
# P90 = P[:,:,1]
# P45 = P[:,:,2]
# P135 = P[:,:,3]

# #Test on Simulated Data 
# title = r'Simulated $45^\circ$'
# P = W_ULTRASIP@Stokes_ideal[:,2]
# P0 = P[:,:,0]
# P90 = P[:,:,1]
# P45 = P[:,:,2]
# P135 = P[:,:,3]

# #Test on Simulated Data 
# title = r'Simulated $135^\circ$'
# P = W_ULTRASIP@Stokes_ideal[:,3]
# P0 = P[:,:,0]
# P90 = P[:,:,1]
# P45 = P[:,:,2]
# P135 = P[:,:,3]

#Test on Simulated Data 
# title = r'Simulated Unpolarized'
# P = W_ULTRASIP@Stokes_ideal[:,4]
# P0 = P[:,:,0]
# P90 = P[:,:,1]
# P45 = P[:,:,2]
# P135 = P[:,:,3]

#Test on measurements
#Horizontal
cal_path = 'D:/Calibration/Data'
# generator_file = glob.glob(f'{cal_path}/Malus*1118_*.h5')
# g = h5py.File(generator_file[3],'r+')
# print(g)
# # #Horizontal Measurements generator, analyzer
# runs = g["Measurement_Metadata"].attrs['Runs for each angle']
# gang = g["Measurement_Metadata"].attrs['Angle of Generator Linear Polarizer']
# raw_0 = np.mean(g["P_0 Measurements/UV Raw Images"][:].reshape(runs,2848,2848),axis=0)
# raw_90 = np.mean(g["P_90 Measurements/UV Raw Images"][:].reshape(runs,2848,2848),axis=0)
# raw_45 = np.mean(g["P_45 Measurements/UV Raw Images"][:].reshape(runs,2848,2848),axis=0)
# raw_135 = np.mean(g["P_135 Measurements/UV Raw Images"][:].reshape(runs,2848,2848),axis=0)

# if gang == '0': 
#     title = 'Measured Horizontal'
# if gang == '90':
#     title = 'Measured Vertical'
# if gang == '45':
#     title = r'Measured $45^\circ$'
# if gang == '135':
#     title = r'Measured $135^\circ$'

#Unpolarized
title = 'Measured Unpolarized'
generator_file = glob.glob(f'{cal_path}/NUC_*.h5')
g = h5py.File(generator_file[7],'r+')
print(g)
exp_times = g['P_0 Measurements/Exposure Times'][:]
Nexp = len(exp_times)
raw_0 =   g["P_0 Measurements/UV Raw Images"][:].reshape(Nexp,2848,2848)
raw_90 =  g["P_90 Measurements/UV Raw Images"][:].reshape(Nexp,2848,2848)
raw_45 =  g["P_45 Measurements/UV Raw Images"][:].reshape(Nexp,2848,2848)
raw_135 = g["P_135 Measurements/UV Raw Images"][:].reshape(Nexp,2848,2848)

r_0 = raw_0[Nexp-5,:,:]
r_90 = raw_90[Nexp-5,:,:]
r_45 = raw_45[Nexp-5,:,:]
r_135 = raw_135[Nexp-5,:,:]
print(exp_times[Nexp-5])

#NUC Images
P0 = correct_img(r_0,Rij[0],Bij[0])
P90 = correct_img(r_90,Rij[90],Bij[90])
P45 = correct_img(r_45,Rij[45],Bij[45])
P135 = correct_img(r_135,Rij[135],Bij[135])


cmin = 200
cmax= 3000
fig = plt.figure(figsize=(16, 8))

# -------- Row 1: Images --------
plt.subplot(2,4,1)
plt.title("P0")
plt.imshow(P0, cmap='gray', interpolation='None', vmin=cmin, vmax=cmax)
plt.colorbar(shrink=0.5)

plt.subplot(2,4,2)
plt.title("P90")
plt.imshow(P90, cmap='gray', interpolation='None', vmin=cmin, vmax=cmax)
plt.colorbar(shrink=0.5)

plt.subplot(2,4,3)
plt.title("P45")
plt.imshow(P45, cmap='gray', interpolation='None', vmin=cmin, vmax=cmax)
plt.colorbar(shrink=0.5)

plt.subplot(2,4,4)
plt.title("P135")
plt.imshow(P135, cmap='gray', interpolation='None', vmin=cmin, vmax=cmax)
plt.colorbar(shrink=0.5)

# -------- Row 2: Histograms with std --------
plt.subplot(2,4,5)
plt.hist(P0.flatten(), bins=100, range=(cmin, cmax))
std0 = np.std(P0)
plt.text(0.95, 0.95, f"std = {std0:.2f}", 
         ha='right', va='top', transform=plt.gca().transAxes)
plt.title("Hist P0")

plt.subplot(2,4,6)
plt.hist(P90.flatten(), bins=100, range=(cmin, cmax))
std90 = np.std(P90)
plt.text(0.95, 0.95, f"std = {std90:.2f}", 
         ha='right', va='top', transform=plt.gca().transAxes)
plt.title("Hist P90")

plt.subplot(2,4,7)
plt.hist(P45.flatten(), bins=100, range=(cmin, cmax))
std45 = np.std(P45)
plt.text(0.95, 0.95, f"std = {std45:.2f}", 
         ha='right', va='top', transform=plt.gca().transAxes)
plt.title("Hist P45")

plt.subplot(2,4,8)
plt.hist(P135.flatten(), bins=100, range=(cmin, cmax))
std135 = np.std(P135)
plt.text(0.95, 0.95, f"std = {std135:.2f}", 
         ha='right', va='top', transform=plt.gca().transAxes)
plt.title("Hist P135")

fig.suptitle(title, fontsize=18, y=0.98)
plt.tight_layout()
plt.show()


P = np.stack([P0, P90, P45, P135], axis=-1)

# Apply per-pixel calibration
# Stokes = pinv(W) @ P_per_pixel
Stokes = np.matmul(pinvW_ULTRASIP, P[...,None])[...,0]       # shape = (H, W, 3)


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
aolp = np.mod(aolp,180)
aolp_std = np.std(aolp)

fig = plt.figure(figsize=(15, 8))

# -------- Row 1: Images --------
plt.subplot(2,3,1)
plt.title("I")
plt.imshow(I, cmap='gray', interpolation='None',vmin=1.75,vmax=2)
plt.colorbar(shrink=0.75)

plt.subplot(2,3,2)
plt.title("Q/I")
plt.imshow(Q/I, cmap=cmo.curl, interpolation='None', vmin=-.1, vmax=.1)
plt.colorbar(shrink=0.75)

plt.subplot(2,3,3)
plt.title("U/I")
plt.imshow(U/I, cmap=cmo.curl, interpolation='None', vmin=-.1, vmax=.1)
plt.colorbar(shrink=0.75)

# -------- Row 2: Histograms with std --------
plt.subplot(2,3,4)
plt.hist(I.flatten(), bins=100)
std_I = np.std(I)
plt.text(0.95, 0.95, f"std = {std_I:.3f}",
         ha='right', va='top', transform=plt.gca().transAxes)
plt.title("Hist I")

plt.subplot(2,3,5)
plt.hist((Q/I).flatten(), bins=100, range=(-0.3,0.3))
std_QI = np.std(Q/I)
plt.text(0.95, 0.95, f"std = {std_QI:.3f}",
         ha='right', va='top', transform=plt.gca().transAxes)
plt.title("Hist Q/I")

plt.subplot(2,3,6)
plt.hist((U/I).flatten(), bins=100, range=(-0.3,0.3))
std_UI = np.std(U/I)
plt.text(0.95, 0.95, f"std = {std_UI:.3f}",
         ha='right', va='top', transform=plt.gca().transAxes)
plt.title("Hist U/I")

plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(8, 6))

counts, bins, patches = ax.hist(
    aolp.flatten(),
    
    bins=100
)

ax.set_title('AoLP  [$\\circ$]', fontsize=16)
ax.tick_params(axis='both', labelsize=14)

ax.xaxis.set_major_locator(MultipleLocator(15))
ax.xaxis.set_minor_locator(MultipleLocator(2))
# Find bins that actually contain data
nonzero = np.where(counts > 0)[0]

if len(nonzero) > 0:

    # index of min and max occupied bins
    i_min = nonzero[0]
    i_max = nonzero[-1]

    # index of highest-frequency bin
    i_peak = np.argmax(counts)

    # unique set of bins to label (avoids duplicates if peak = min or max)
    idx_to_label = sorted({i_min, i_max, i_peak})

    for i in idx_to_label:
        bin_center = (bins[i] + bins[i+1]) / 2

        ax.text(
            bin_center,
            counts[i],
            f'{bin_center:.1f}°',
            ha='center', va='bottom',
            fontsize=10,
            rotation=0
        )
        
        
# Place a text box on the figure
textstr = f"Std = {aolp_std:.2f}°"

ax.text(
    0.97, 0.95, textstr,
    transform=ax.transAxes,          # coordinates in figure space (0–1)
    ha='right', va='top',
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.4", 
              edgecolor="black", 
              facecolor="white",
              alpha=0.8)
)

plt.show()

plt.figure()
plt.imshow(dolp, cmap='hot', interpolation = 'None',vmin=80,vmax=100)
plt.title('DoLP [%]',fontsize=20)
plt.colorbar()

plt.figure()
plt.imshow(aolp, cmap=cmo.phase, interpolation = 'None',vmin=0,vmax=180)
plt.title('AoLP  [deg]',fontsize=20)
plt.colorbar()

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(dolp.flatten(),range=(0,10),bins=100)
ax.set_title('DoLP  [%]',fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.xaxis.set_major_locator(MultipleLocator(1))   
ax.xaxis.set_minor_locator(MultipleLocator(0.1))   
# Add text box to the right of plot
textstr = f"Mean = {dolp_mean:.4f}%\nStd = {dolp_std:.4f}%\nMed = {dolp_median:.4f}%"
ax.text(0.5, 0.8, textstr, transform=ax.transAxes, fontsize=14,
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
