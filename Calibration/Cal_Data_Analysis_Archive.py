# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 10:57:23 2025

@author: ULTRASIP_1
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 16:04:35 2025

@author: Clarissa DeLeon 

This code is to generate plots of the calibration data captured using 
'Cal_Data_Acquisition'.

Specific analysis is given under the different calibration types. 

More info in ULTRASIP User Manual: 
    https://www.overleaf.com/read/hkkghcvgrrdt#c1c7d8
"""

#Import libraries 
import os
import time
import h5py
import glob
import serial 
import numpy as np
import cmocean.cm as cmo
from datetime import datetime 
import matplotlib.pyplot as plt
from scipy.stats import linregress


#Load data file
#Define data path
data_path = "D:/ULTRASIP/Calibration/Cal_Data"
file = glob.glob(f'{data_path}/Lab*2025_04_16_13*.h5') #add name of file otherwise it will load all .h5 files
f = h5py.File(file[0],'r+') #set file to read mode
Cal_Type = f['Measurement_Metadata'].attrs['Cal Type']

if Cal_Type == 'Flat_Field':

    print('flat')
    
    # for ang in range(0,360,15):
        
    #     uvimg = f[f'UV Image P{ang} Data'][f'UV Images P{ang}'][:]
    #     uvimg = uvimg.reshape(10,2848,2848)
    
    #     #Average image
    #     plt.figure()
    #     avg_uvimg = np.average(uvimg,axis=0)
    #     plt.imshow(avg_uvimg, interpolation  = 'None', cmap='gray')
    #     plt.title(f'P{ang} Average Flat Field')
    #     plt.colorbar()
    
    #     plt.figure()
    #     plt.hist(avg_uvimg.flatten())
    #     plt.xlabel('Pixel Value')
    #     plt.ylabel('Number of Pixels')
    
    #     print(f'P{ang}',np.std(avg_uvimg))
    #     print(f'P{ang}',np.average(avg_uvimg))
    
        
    uvimg0 = f['UV Image P0 Data']['UV Images P0'][:]
    uvimg0 = uvimg0.reshape(10,2848,2848)
    
    #Average image
    plt.figure()
    avg_uvimg0 = np.average(uvimg0,axis=0)
    plt.imshow(avg_uvimg0, interpolation  = 'None', cmap='gray')
    plt.title('P0')
    plt.colorbar()
    
    plt.figure()
    plt.hist(avg_uvimg0.flatten())
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of Pixels')
    
    print('P0',np.std(avg_uvimg0))
    print('P0',np.average(avg_uvimg0))
    
    
    uvimg45 = f['UV Image P45 Data']['UV Images P45'][:]
    uvimg45 = uvimg45.reshape(10,2848,2848)
    
    #Average image
    plt.figure()
    avg_uvimg45 = np.average(uvimg45,axis=0)
    plt.title('P45')
    plt.imshow(avg_uvimg45, interpolation  = 'None', cmap='gray')
    plt.colorbar()
    
    plt.figure()
    plt.hist(avg_uvimg45.flatten())
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of Pixels')
    
    print('P45',np.std(avg_uvimg45))
    print('P45',np.average(avg_uvimg45))

    uvimg90 = f['UV Image P90 Data']['UV Images P90'][:]
    uvimg90 = uvimg90.reshape(10,2848,2848)
    
    #Average image
    plt.figure()
    avg_uvimg90 = np.average(uvimg90,axis=0)
    plt.title('P90')
    plt.imshow(avg_uvimg90, interpolation  = 'None', cmap='gray')
    plt.colorbar()
    
    plt.figure()
    plt.hist(avg_uvimg90.flatten())
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of Pixels')
    
    print('P90',np.std(avg_uvimg90))
    print('P90',np.average(avg_uvimg90))

    uvimg135 = f['UV Image P135 Data']['UV Images P135'][:]
    uvimg135 = uvimg135.reshape(10,2848,2848)
    
    #Average image
    plt.figure()
    avg_uvimg135 = np.average(uvimg135,axis=0)
    plt.title('P135')
    plt.imshow(avg_uvimg135, interpolation  = 'None', cmap='gray')
    plt.colorbar()
    
    plt.figure()
    plt.hist(avg_uvimg135.flatten())
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of Pixels')
    
    print('P135',np.std(avg_uvimg135))
    print('P135',np.average(avg_uvimg135))
    
    I = avg_uvimg0 + avg_uvimg90
    Q = avg_uvimg0 - avg_uvimg90
    U = avg_uvimg45 - avg_uvimg135
    
    dolp = (np.sqrt((Q**2)+(U**2))/I)
    
    aolp = np.degrees(0.5*np.arctan2(U,Q))
    aolp = np.mod(aolp, 180)
    
        
    plt.figure()
    plt.imshow(Q/I, cmap='coolwarm',
    interpolation='None')
    plt.title('Q/I')
    plt.colorbar()
    
    plt.figure()
    plt.imshow(U/I, cmap='coolwarm',
    interpolation='None')
    plt.title('U/I')
    plt.colorbar()

    plt.figure()
    plt.imshow(aolp, interpolation  = 'None', vmin=0,vmax=180, cmap=cmo.phase)
    plt.title('AoLP [deg]')
    plt.colorbar()
    
    plt.figure()
    plt.hist(aolp.flatten())
    plt.xlabel('AoLP [deg]')
    plt.ylabel('Number of Pixels')
    
    plt.figure()
    plt.hist(dolp.flatten()*100)
    plt.xlabel('DoLP [%]')
    plt.ylabel('Number of Pixels')
    
    plt.figure()
    plt.imshow(np.log(dolp), cmap='Blues_r',
    interpolation='None', vmin=-7, vmax=-4)
    plt.title('log(dolp)')
    plt.colorbar()

    # Plot
    plt.figure()
    plt.plot(np.arange(avg_uvimg0.shape[1]),np.average(avg_uvimg0/I, axis=0), color='black',label='P0/I Column avg')
    plt.plot(np.arange(avg_uvimg0.shape[0]),np.average(avg_uvimg0/I, axis=1), color = 'red', label = 'P0/I Row avg')
    plt.xlabel('Pixel Index')
    plt.ylabel('Digital Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot
    plt.figure()
    plt.plot(np.arange(avg_uvimg45.shape[1]),np.average(avg_uvimg45/I, axis=0), color='black',label='P45/I Column avg')
    plt.plot(np.arange(avg_uvimg45.shape[0]),np.average(avg_uvimg45/I, axis=1), color = 'red', label = 'P45/I Row avg')
    plt.xlabel('Pixel Index')
    plt.ylabel('Digital Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot
    plt.figure()
    plt.plot(np.arange(avg_uvimg90.shape[1]),np.average(avg_uvimg90/I, axis=0), color='black',label='P90/I Column avg')
    plt.plot(np.arange(avg_uvimg90.shape[0]),np.average(avg_uvimg90/I, axis=1), color = 'red', label = 'P90/I Row avg')
    plt.xlabel('Pixel Index')
    plt.ylabel('Digital Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot
    plt.figure()
    plt.plot(np.arange(avg_uvimg135.shape[1]),np.average(avg_uvimg135/I, axis=0), color='black',label='P135/I Column avg')
    plt.plot(np.arange(avg_uvimg135.shape[0]),np.average(avg_uvimg135/I, axis=1), color = 'red', label = 'P135/I Row avg')
    plt.xlabel('Pixel Index')
    plt.ylabel('Digital Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot
    plt.figure()
    plt.plot(np.arange(avg_uvimg0.shape[1]),np.average(Q/I, axis=0), color='black',label='Q/I Column avg')
    plt.plot(np.arange(avg_uvimg0.shape[0]),np.average(Q/I, axis=1), color = 'red', label = 'Q/I Row avg')
    plt.xlabel('Pixel Index')
    plt.ylabel('Digital Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot
    plt.figure()
    plt.plot(np.arange(avg_uvimg0.shape[1]),np.average(U/I, axis=0), color='black',label='U/I Column avg')
    plt.plot(np.arange(avg_uvimg0.shape[0]),np.average(U/I, axis=1), color = 'red', label = 'U/I Row avg')
    plt.xlabel('Pixel Index')
    plt.ylabel('Digital Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
if Cal_Type == 'Exposure_Set':

    print('exposure')
    
    # Paths to your HDF5 image groups and the corresponding exposure values
    image_paths = [
        ('UV Image 87222 Data', 'UV Images 87222', 87222),
        ('UV Image 70000 Data', 'UV Images 70000', 70000),
        ('UV Image 156111 Data', 'UV Images 156111', 156111),
        ('UV Image 173333 Data', 'UV Images 173333', 173333),
        ('UV Image 190555 Data', 'UV Images 190555', 190555),
        ('UV Image 104444 Data', 'UV Images 104444', 104444),
        ('UV Image 121666 Data','UV Images 121666', 121666),
        ('UV Image 138888 Data','UV Images 138888', 138888)
        ]

    # Open your HDF5 file
    with f as f:
        images = []
        exposure_times = []

        for group, dataset, exposure in image_paths:
            uvimg = f[group][dataset][:]
            uvimg = uvimg.reshape(10, 2848, 2848)
            uvimg_avg = np.mean(uvimg, axis=0)
            images.append(uvimg_avg)
            exposure_times.append(exposure)
            plt.figure()
            plt.imshow(uvimg_avg, interpolation='None', cmap='gray', vmin=0, vmax=3000)
            plt.colorbar()
            plt.title(f'P$_{{ij}}$,{exposure} [us]')
            
            # Convert to NumPy arrays
            stacked = np.stack(images, axis=0)  # Shape: (num_exposures, 2848, 2848)
            exposure_times = np.array(exposure_times)
            
            i_range = range(0, 2848)
            j_range = range(0, 2848)
            
            Rij = np.zeros((len(i_range), len(j_range)))
            Bij = np.zeros((len(i_range), len(j_range)))
            
            for i_idx, i in enumerate(i_range):
                for j_idx, j in enumerate(j_range):
                    pixel_values = stacked[:, i, j]
                    slope, intercept = np.polyfit(exposure_times, pixel_values, 1)
                    Rij[i_idx, j_idx] = slope
                    Bij[i_idx, j_idx] = intercept
                    
                    # Get global slope/intercept
                    R_avg = np.mean(Rij)
                    B_avg = np.mean(Bij)
                    
                    # Pick a target image to correct (for visualization, e.g. last one)
                    img = stacked[5, 0:2848, 0:2848]
                    uv_exp = exposure_times[5]
                    # Apply NUC correction
                    Cij = (R_avg / Rij) * (img - Bij) + B_avg
                    
                    plt.figure()
                    plt.scatter(exposure_times, pixel_values)
                    plt.xlabel('Exposure Times (us)')
                    plt.ylabel('Pixel Values')

        plt.figure()
        plt.imshow(Rij, interpolation='None', cmap='gray')
        plt.colorbar()
        plt.title('R$_{ij}$')

        plt.figure()
        plt.imshow(Bij, interpolation='None', cmap='gray')
        plt.colorbar()
        plt.title('B$_{ij}$')

        plt.figure()
        plt.imshow(Cij, interpolation='None', cmap='gray', vmin=0, vmax=3000)
        plt.colorbar()
        plt.title(f'C$_{{ij}}$,{exposure} [us]')

        plt.show()
        
        nuc = f.create_group('NUC Data')
        nuc.create_dataset('Rij', data=Rij)
        nuc.create_dataset('Bij', data=Bij)
