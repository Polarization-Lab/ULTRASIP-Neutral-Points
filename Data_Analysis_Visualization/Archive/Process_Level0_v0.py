# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 21:47:55 2025

@author: C.M.DeLeon

Process 0 - output corrected polarized data products
"""

# Import Libraries
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import numpy as np
import h5py 
import glob
import os, time

# Define W-matrix of ULTRASIP (rows = analyzer vectors P0, P90, P45, P135)
W = 0.5 * np.array([[1, 1, 0],
                    [1, -1, 0],
                    [1, 0, 1],
                    [1, 0, -1]])

# Set Date of Measurements
date = '2025_06_10'

# Data path
start_time = time.time()

#Load Calibration Files 
#Calibration path and file
cal_path = 'D:/Data/2025_08_12/' #DO NOT CHANGE
cal_file = glob.glob(f'{cal_path}/NUC*16_4*.h5')
f = h5py.File(cal_file[0],'r+')
Rij0 = f['NUC Images/P0 Rij'][:]
Bij0 = f['NUC Images/P0 Bij'][:]

Rij45 = f['NUC Images/P45 Rij'][:]
Bij45 = f['NUC Images/P45 Bij'][:]

Rij90 = f['NUC Images/P90 Rij'][:]
Bij90 = f['NUC Images/P90 Bij'][:]

Rij135 = f['NUC Images/P135 Rij'][:]
Bij135 = f['NUC Images/P135 Bij'][:]

#basepath = 'C:/Users/ULTRASIP_1/OneDrive/Desktop/'
basepath = 'D:/Data/'
folderdate = os.path.join(basepath, date)
files = glob.glob(f'{folderdate}/*.h5')
idx = len(files) 
idx_array = np.arange(0,idx)

def process0(idx):
    print(f'Processing file {idx}: {files[idx]}')
    
    try:
        f = h5py.File(files[idx], 'r+')
        
        for aqnum in range(0, len(f.keys()) - 1):
            try:
                # Set acquisition to view
                aq = f[f'Aquistion_{aqnum}']

                # Reconstruct the 4 flux images
                uv_data = aq['UV Image Data']
                uv_imgs = uv_data['UV Raw Images'][:].reshape(4, 2848, 2848)
                P0, P45, P90, P135 = uv_imgs
                
                # Get global slope/intercept
                R_avg0 = np.mean(Rij0)
                B_avg0 = np.mean(Bij0)

                # Apply NUC correction
                P0_corrected = (R_avg0 / Rij0) * (P0 - Bij0) + B_avg0
                
                # Get global slope/intercept
                R_avg45 = np.mean(Rij45)
                B_avg45 = np.mean(Bij45)

                # Apply NUC correction
                P45_corrected = (R_avg45 / Rij45) * (P45 - Bij45) + B_avg45
                
                # Get global slope/intercept
                R_avg90 = np.mean(Rij90)
                B_avg90 = np.mean(Bij90)

                # Apply NUC correction
                P90_corrected = (R_avg90 / Rij90) * (P90 - Bij90) + B_avg90
                
                # Get global slope/intercept
                R_avg135 = np.mean(Rij135)
                B_avg135 = np.mean(Bij135)

                # Apply NUC correction
                P135_corrected = (R_avg135 / Rij135) * (P135  - Bij135 ) + B_avg135 
                
                P_corrected = np.array([P0_corrected, P90_corrected, P45_corrected, P135_corrected])
                Stokes_meas = np.linalg.pinv(W)@(P_corrected.reshape(4,2848*2848))
                Stokes_meas = Stokes_meas.reshape(3, 2848, 2848)

                I_corrected, Q_corrected, U_corrected = Stokes_meas


                # Overwrite or create datasets
                for name in ['I_corrected', 'Q_corrected', 'U_corrected']:
                    if name in uv_data:
                        del uv_data[name]

                uv_data.create_dataset('I_corrected', data=I_corrected)
                uv_data.create_dataset('Q_corrected', data=Q_corrected)
                uv_data.create_dataset('U_corrected', data=U_corrected)

            
            except KeyError as e:
                print(f'Skipping Acquisition {aqnum} in file {idx} — {e}')
                continue
    finally: 
        
        f['Measurement_Metadata'].attrs['Processed Level'] = 'Level 0'
        f.close()
    
threads = cpu_count()

with ThreadPool(threads) as p:
    p.map(process0, idx_array)
       
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")