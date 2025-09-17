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

#Define W-matrix of ULTRASIP 
#The analyzer vectors (P0,P90,P45,P135) are the rows of the W-matrix (pg 230 of PL&OS)
W = 0.5*np.array([[1,1,0],[1,-1,0],[1,0,1],[1,0,-1]])

#Image Size 
Ni = 2848
Nj = 2848

#Datapath
cal_path = 'E:/2025_06_29/' #DO NOT CHANGE
cal_file = glob.glob('E:/Calibration/2025_08_12/NUC*16*.h5')
Calibration_Type = 'NUC'
meas_file = glob.glob(f'{cal_path}/NUC*19_54*.h5')
#files = glob.glob(f'{folderdate}/{Calibration_Type}*11_*.h5')
idx = len(cal_file)-1 # Set file index you want to view - default is set to the last one (len(files)-1)
f = h5py.File(cal_file[idx],'r+')
m= h5py.File(meas_file[idx],'r+')


#exp = f['Measurement_Metadata/P_0 Measurements/Exposure Times'][:]

exp = m['Measurement_Metadata/P_0 Measurements/Exposure Times'][:]

uvimgs0 = m['Measurement_Metadata/P_0 Measurements/UV Raw Images'][:]
P0_meas = uvimgs0.reshape(len(exp),Ni,Nj)

uvimgs45 = m['Measurement_Metadata/P_45 Measurements/UV Raw Images'][:]
P45_meas = uvimgs45.reshape(len(exp),Ni,Nj)

uvimgs90 = m['Measurement_Metadata/P_90 Measurements/UV Raw Images'][:]
P90_meas = uvimgs90.reshape(len(exp),Ni,Nj)

uvimgs135 = m['Measurement_Metadata/P_135 Measurements/UV Raw Images'][:]
P135_meas = uvimgs135.reshape(len(exp),Ni,Nj)

R0ij = f['NUC Images/P0 Rij']
B0ij = f['NUC Images/P0 Bij']

R45ij = f['NUC Images/P45 Rij']
B45ij = f['NUC Images/P45 Bij']

R90ij = f['NUC Images/P90 Rij']
B90ij = f['NUC Images/P90 Bij']

R135ij = f['NUC Images/P135 Rij']
B135ij = f['NUC Images/P135 Bij']

# Get global slope/intercept
R0_avg = np.mean(R0ij)
B0_avg = np.mean(B0ij)

P0ij = P0_meas[39,:,:]
# Apply NUC correction
C0ij = (R0_avg / R0ij) * (P0ij - B0ij) + B0_avg


fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
axes[0].imshow(P0ij, interpolation='None', cmap='gray',vmin=0,vmax=2500)
axes[0].set_title('Original Image P0$_{ij}$',fontsize=20)

im1=axes[1].imshow(C0ij, interpolation='None', cmap='gray',vmin=0,vmax=2500)
axes[1].set_title('Corrected Image C0$_{ij}$',fontsize=20)

plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Get global slope/intercept
R45_avg = np.mean(R45ij)
B45_avg = np.mean(B45ij)

P45ij = P45_meas[39,:,:]
# Apply NUC correction
C45ij = (R45_avg / R45ij) * (P45ij - B45ij) + B45_avg

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
axes[0].imshow(P45ij, interpolation='None', cmap='gray',vmin=0,vmax=2500)
axes[0].set_title('Original Image P45$_{ij}$',fontsize=20)

im1=axes[1].imshow(C45ij, interpolation='None', cmap='gray',vmin=0,vmax=2500)
axes[1].set_title('Corrected Image C45$_{ij}$',fontsize=20)

plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Get global slope/intercept
R90_avg = np.mean(R90ij)
B90_avg = np.mean(B90ij)

P90ij = P90_meas[39,:,:]
# Apply NUC correction
C90ij = (R90_avg / R90ij) * (P90ij - B90ij) + B90_avg

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
axes[0].imshow(P90ij, interpolation='None', cmap='gray',vmin=0,vmax=2500)
axes[0].set_title('Original Image P90$_{ij}$',fontsize=20)

im1=axes[1].imshow(C90ij, interpolation='None', cmap='gray',vmin=0,vmax=2500)
axes[1].set_title('Corrected Image C90$_{ij}$',fontsize=20)

plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Get global slope/intercept
R135_avg = np.mean(R135ij)
B135_avg = np.mean(B135ij)

P135ij = P135_meas[39,:,:]
# Apply NUC correction
C135ij = (R135_avg / R135ij) * (P135ij - B135ij) + B135_avg

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
axes[0].imshow(P135ij, interpolation='None', cmap='gray',vmin=0,vmax=2500)
axes[0].set_title('Original Image P135$_{ij}$',fontsize=20)

im1=axes[1].imshow(C135ij, interpolation='None', cmap='gray',vmin=0,vmax=2500)
axes[1].set_title('Corrected Image C135$_{ij}$',fontsize=20)

plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

P = np.array([P0ij,P90ij,P45ij,P135ij])

Stokes_meas = np.linalg.pinv(W)@(P.reshape(4,2848*2848))
Stokes_meas = Stokes_meas.reshape(3,2848,2848)

I = Stokes_meas[0,:,:]
Q = Stokes_meas[1,:,:]/I[:]
U = Stokes_meas[2,:,:]/I[:]
    
dolp = np.sqrt((Q**2)+(U**2))*100
    
aolp = 0.5*np.arctan2(U,Q)
aolp = np.mod(np.degrees(aolp),180)

plt.figure()
plt.imshow(dolp, cmap='hot', interpolation = 'None',vmin=0,vmax=1)
plt.title('DolP OG [%]',fontsize=20)
plt.colorbar()

plt.figure()
plt.hist(dolp.flatten())
plt.title('DoLP OG [%]')

plt.figure()
plt.imshow(aolp, cmap=cmo.phase, interpolation = 'None')
plt.title('AoLP OG [deg]',fontsize=20)
plt.colorbar()

plt.figure()
plt.hist(aolp.flatten())
plt.title('AoLP OG [deg]')
plt.show()


C = np.array([C0ij,C90ij,C45ij,C135ij])

Stokes_measc = np.linalg.pinv(W)@(C.reshape(4,2848*2848))
Stokes_measc = Stokes_meas.reshape(3,2848,2848)

Ic = Stokes_meas[0,:,:]
Qc= Stokes_meas[1,:,:]/Ic[:]
Uc = Stokes_meas[2,:,:]/Ic[:]

#Axis 0 is column average, Axis 1 is row average
row_avgQc = np.average(Qc, axis=1)
col_avgQc = np.average(Qc, axis=0)

row_avgUc = np.average(Uc, axis=1)
col_avgUc = np.average(Uc, axis=0)

    
dolpc = np.sqrt((Q**2)+(U**2))*100
    
aolpc = 0.5*np.arctan2(U,Q)
aolpc = np.mod(np.degrees(aolp),180)

plt.figure()
plt.imshow(aolpc, cmap=cmo.phase, interpolation = 'None')
plt.title('AoLP Corrected [deg]',fontsize=20)
plt.colorbar()
plt.show()

plt.figure()
plt.hist(aolpc.flatten())
plt.title('AoLP Corrected [deg]')
plt.show()

plt.figure()
plt.imshow(dolpc, cmap='hot', interpolation = 'None',vmin=0,vmax=1)
plt.title('DoLP Corrected [%]',fontsize=20)
plt.colorbar()
plt.show()

plt.figure()
plt.hist(dolpc.flatten())
plt.title('DoLP Corrected [%]')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im1=axes[0].imshow(Qc, interpolation='None', cmap=cmo.curl,vmin=-0.075,vmax=0.075)
axes[0].set_title('Corrected Image Q/I',fontsize=20)
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)


im1=axes[1].scatter(row_avgQc,range(0,len(row_avgQc)),color='green')
axes[1].set_title('Row Average',fontsize=20)

im1=axes[2].scatter(col_avgQc,range(0,len(col_avgQc)),color='green')
axes[2].set_title('Column Average',fontsize=20)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im1=axes[0].imshow(Uc, interpolation='None', cmap=cmo.curl,vmin=-0.075,vmax=0.075)
axes[0].set_title('Corrected Image U/I',fontsize=20)
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im1=axes[1].scatter(row_avgUc,range(0,len(row_avgUc)),color='green')
axes[1].set_title('Row Average',fontsize=20)

im1=axes[2].scatter(col_avgUc,range(0,len(col_avgUc)),color='green')
axes[2].set_title('Column Average',fontsize=20)

plt.tight_layout()
plt.show()

print('Average DoLP',np.average(dolpc), 'Standard Deviation', np.std(dolpc))

dolp_rc = np.sqrt(row_avgQc**2 + col_avgUc**2)*100
print('Average DoLPrc',np.average(dolp_rc), 'Standard Deviation rc', np.std(dolp_rc))

plt.figure()
plt.scatter(range(0,len(dolp_rc)),dolp_rc,color='green')
plt.ylabel(r'$DoLP_{rc} [\%]$',fontsize=15)
plt.xlabel('Pixel Index',fontsize=15)
plt.title(r'DoLP from $\bar{c}_{U},\bar{r}_{Q}$')
plt.show()
