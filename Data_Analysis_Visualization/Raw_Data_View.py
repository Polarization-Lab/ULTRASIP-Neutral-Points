# -*- coding: utf-8 -*-
"""
@author: C.M.DeLeon

ULTRASIP User Manual: 
    https://www.overleaf.com/read/hkkghcvgrrdt#c1c7d8


DataView:
    View raw data as a first check
"""

#Libraries 
import numpy as np
import h5py
import os
import glob
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from matplotlib.widgets import Slider, TextBox

#Define W-matrix of ULTRASIP 
#The analyzer vectors (P0,P90,P45,P135) are the rows of the W-matrix (pg 230 of PL&OS)
W = 0.5*np.array([[1,1,0],[1,-1,0],[1,0,1],[1,0,-1]])

#Load observations 
#Set Date of Measurements 
date = '2025_07_21'

#Datapath
basepath = 'D:/Data'
#basepath = 'C:/Users/ULTRASIP_1/OneDrive/Desktop'
folderdate = os.path.join(basepath,date)
files = glob.glob(f'{folderdate}/*.h5')
idx = len(files)-1 # Set file index you want to view - default is set to the last one (len(files)-1)
f = h5py.File(files[idx],'r+')
print(files[idx])

for aqnum in range(0,len(f.keys())-1):
    
    timestamp = files[idx][38:46]
    #Set aquisition to view
    aq = f[f'Aquistion_{aqnum}']
    
    print(aqnum)
    
    #Reconstruct the 4 flux images and calculate products
    uv_imgs = aq['UV Image Data/UV Raw Images'][:]
    uv_imgs = uv_imgs.reshape(4,2848,2848)
    P0 = uv_imgs[0,:,:]
    P45 = uv_imgs[1,:,:]
    P90 = uv_imgs[2,:,:]
    P135 = uv_imgs[3,:,:]

    
    P = np.array([P0,P90,P45,P135])

    Stokes_meas = np.linalg.pinv(W)@(P.reshape(4,2848*2848))
    Stokes_meas = Stokes_meas.reshape(3,2848,2848)

    I = Stokes_meas[0,:,:]
    Q = Stokes_meas[1,:,:]/I[:]
    U = Stokes_meas[2,:,:]/I[:]
    
    #axis 0 is along columns, axis=1 is along rows
    avgQ = np.flip(np.average(Q, axis=1))
    avgU = np.average(U, axis=0)
    
    dolp = np.sqrt((Q**2)+(U**2))*100
    
    aolp = 0.5*np.arctan2(U,Q)
    aolp = np.mod(np.degrees(aolp),180)
    
    #------------------------Plots------------------------------------------#
    
    #Polarized flux
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True)

    vmin, vmax = 1000, 4200
    cmap = 'gray'

    im = axes[0].imshow(P0, cmap=cmap, vmin=vmin, vmax=vmax, interpolation = 'None')
    axes[0].set_title('P0',fontsize=20)

    axes[1].imshow(P90, cmap=cmap, vmin=vmin, vmax=vmax, interpolation = 'None')
    axes[1].set_title('P90',fontsize=20)

    axes[2].imshow(P45, cmap=cmap, vmin=vmin, vmax=vmax, interpolation = 'None')
    axes[2].set_title('P45',fontsize=20)

    axes[3].imshow(P135, cmap=cmap, vmin=vmin, vmax=vmax, interpolation = 'None')
    axes[3].set_title('P135',fontsize=20)

    fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, wspace=0.3)

    cbar_ax = fig.add_axes([0.2, 0.88, 0.6, 0.04])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=20)
    plt.suptitle(f'{timestamp, aqnum}',fontsize=20)
    plt.show()
    

    #Stokes parameters

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # 1 row, 4 columns

    # Plot I histogram
    im0 = axes[0].hist(I.flatten(), bins=30, color='blue')
    axes[0].set_title('I Histogram',fontsize=20)
    
    #Plot I
    im1 = axes[1].imshow(I, cmap='gray', vmin=0, vmax=8000, interpolation = 'None')
    axes[1].set_title('I',fontsize=20)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot Q/I
    im2 = axes[2].imshow(Q, cmap=cmo.curl, interpolation = 'None')
    axes[2].set_title('Q/I',fontsize=20)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Plot U/I
    im3 = axes[3].imshow(U, cmap=cmo.curl, interpolation = 'None')
    axes[3].set_title('U/I',fontsize=20)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    plt.suptitle(f'{timestamp, aqnum}',fontsize=20)

    plt.tight_layout()
    plt.show()
    
    #Averages
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    axes[0].scatter(avgQ,range(0,len(avgQ)),color='green')
    axes[0].set_title('$r_{Q}$',fontsize=20)
    axes[0].axvline(x=0,lw=5,color='red')
    axes[0].set_xlim(-0.025, 0.025)
    
    axes[1].scatter(avgU,range(0,len(avgU)),color='green')
    axes[1].set_title('$c_{U}$',fontsize=20)
    axes[1].axvline(x=0,lw=5,color='red')
    axes[1].set_xlim(-0.025, 0.025)
    
    plt.tight_layout()
    plt.show()

    fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, wspace=0.3)

    plt.suptitle(f'{timestamp, aqnum}',fontsize=20)
    plt.show()
    
    #DoLP/AoLP

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # 1 row, 4 columns

    # Plot DoLP
    im0 = axes[0].imshow(dolp, cmap='hot', interpolation = 'None', vmin = 0, vmax = 2)
    axes[0].set_title('DoLP [%]',fontsize=20)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    #Plot log(DoLP)
    im1 = axes[1].imshow(np.log(dolp), cmap='Blues_r', vmin=-2, vmax=2, interpolation = 'None')
    axes[1].set_title('log(DoLP [%])',fontsize=20)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot AoLP
    im2 = axes[2].imshow(aolp, cmap=cmo.phase, interpolation = 'None')
    axes[2].set_title('AoLP [deg]',fontsize=20)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Plot AoLP
    im3 = axes[3].hist(aolp.flatten())
    axes[3].set_title('AoLP [deg]',fontsize=20)
    plt.suptitle(f'{timestamp, aqnum}',fontsize=20)

    plt.tight_layout()
    plt.show()
    


# Close the HDF5 file
f.close()
