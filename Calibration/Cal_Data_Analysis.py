# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 19:55:40 2025

@author: C.M.DeLeon
Analyze Calibration Data
"""

#Import libraries 
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import glob
import h5py
import os

#Image Size 
Ni = 2848
Nj = 2848

Rij0 = np.zeros((Ni, Nj))
Bij0 = np.zeros((Ni, Nj))

Rij45 = np.zeros((Ni, Nj))
Bij45 = np.zeros((Ni, Nj))

Rij90 = np.zeros((Ni, Nj))
Bij90 = np.zeros((Ni, Nj))

Rij135 = np.zeros((Ni, Nj))
Bij135 = np.zeros((Ni, Nj))

#Datapath
basepath = 'E:/'
date = '2025_06_29'
Calibration_Type = 'NUC'
folderdate = os.path.join(basepath,date)
files = glob.glob(f'{folderdate}/{Calibration_Type}*19_54*.h5')
idx = len(files)-1 # Set file index you want to view - default is set to the last one (len(files)-1)
f = h5py.File(files[idx],'r+')
exp = f['Measurement_Metadata/P_0 Measurements/Exposure Times'][:]


uvimgs0 = f['Measurement_Metadata/P_0 Measurements/UV Raw Images'][:]
P0_meas = uvimgs0.reshape(len(exp),Ni,Nj)

uvimgs45 = f['Measurement_Metadata/P_45 Measurements/UV Raw Images'][:]
P45_meas = uvimgs45.reshape(len(exp),Ni,Nj)

uvimgs90 = f['Measurement_Metadata/P_90 Measurements/UV Raw Images'][:]
P90_meas = uvimgs90.reshape(len(exp),Ni,Nj)

uvimgs135 = f['Measurement_Metadata/P_135 Measurements/UV Raw Images'][:]
P135_meas = uvimgs135.reshape(len(exp),Ni,Nj)



for i in range(0,Ni):
    for j in range(0,Nj):
        pixel_values0 = P0_meas[:,i,j]
        slope0, intercept0, *_ = linregress(exp, pixel_values0)
        Rij0[i, j] = slope0
        Bij0[i, j] = intercept0

        pixel_values45 = P45_meas[:,i,j]
        slope45, intercept45, *_ = linregress(exp, pixel_values45)
        Rij45[i, j] = slope45
        Bij45[i, j] = intercept45
        
        pixel_values90 = P90_meas[:,i,j]
        slope90, intercept90, *_ = linregress(exp, pixel_values90)
        Rij90[i, j] = slope90
        Bij90[i, j] = intercept90
        
        pixel_values135 = P135_meas[:,i,j]
        slope135, intercept135, *_ = linregress(exp, pixel_values135)
        Rij135[i, j] = slope135
        Bij135[i, j] = intercept135
        
print('saving')   
#Save
nuc = f.create_group("NUC Images")
nuc = f['NUC Images']
nuc.create_dataset('P0 Rij', data = Rij0)
nuc.create_dataset('P45 Rij', data = Rij45)
nuc.create_dataset('P90 Rij', data = Rij90)
nuc.create_dataset('P135 Rij', data = Rij135)

nuc.create_dataset('P0 Bij', data = Bij0)
nuc.create_dataset('P45 Bij', data = Bij45)
nuc.create_dataset('P90 Bij', data = Bij90)
nuc.create_dataset('P135 Bij', data = Bij135)

        
# Get global slope/intercept
R_avg = np.mean(Rij0)
B_avg = np.mean(Bij0)

Pij = P0_meas[10,:,:]
# Apply NUC correction
Cij = (R_avg / Rij0) * (Pij - Bij0) + B_avg


plt.figure()
plt.scatter(exp, pixel_values0, color='blue')
plt.xlabel('Exposure Value')
plt.ylabel(f'Pixel Intensity at ({i},{j})')
plt.title(f'Pixel Response: ({i},{j})')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting
plt.figure()
plt.imshow(Rij0, interpolation='None', cmap='gray')
plt.colorbar()
plt.title('R$_{ij}$')
plt.show()

plt.figure()
plt.imshow(Bij0, interpolation='None', cmap='gray',vmin=0,vmax=10)
plt.colorbar()
plt.title('B$_{ij}$')
plt.show()

plt.figure()
plt.hist(Pij.flatten())
plt.show()

plt.figure()
plt.imshow(Cij, interpolation='None', cmap='gray',vmin=1000,vmax=2500)
plt.colorbar()
plt.title('Corrected Image C$_{ij}$')
plt.show()

plt.figure()
plt.hist(Cij.flatten())
plt.show()