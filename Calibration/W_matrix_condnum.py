# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 21:36:05 2026

@author: deleo
"""

#Import libraries 
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import MultipleLocator
import numpy as np
import cmocean.cm as cmo
import glob
import h5py
import os

def cond_num(W):
    
    # compute singular values for each 4x3 matrix
    u, s, vh = np.linalg.svd(W, full_matrices=False)

    # condition number = largest singular value / smallest singular value
    cond_W = s[..., 0] / s[..., -1]
    return cond_W

def plot_condW(condW,px,py):
    
    fig, ax = plt.subplots(figsize=(7,6))

    im = ax.imshow(condW, interpolation='None',cmap='coolwarm',vmin=1.4,vmax=1.5)

    cbar = fig.colorbar(im, ax=ax,shrink=0.8)
    cbar.set_label('Condition Number', fontsize=14)   
    cbar.ax.tick_params(labelsize=12)                             
    #ax.add_patch(patches.Rectangle((px, py), 5, 5, linewidth=5, edgecolor='green', facecolor='none'))
    ax.set_xlabel('Pixel', fontsize=14)
    ax.set_ylabel('Pixel', fontsize=14)

    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots()

    cond_flat = condW.flatten()
    std_val = np.std(cond_flat)

    ax.hist(cond_flat, range=(1.4, 1.5), bins=100)
    ax.set_ylim([0, 5e4])

    # Add text box with std in upper-right corner of plot
    textstr = f"std = {std_val:.4e}"

    ax.text(0.97, 0.97, textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.show()

Wnone = np.load('D:/ULTRASIP_Wmatrix_None.npy')       
W_15 = np.load('D:/ULTRASIP_Wmatrix_15.npy')       
W_30 = np.load('D:/ULTRASIP_Wmatrix_30.npy')       
W_45 = np.load('D:/ULTRASIP_Wmatrix_45.npy')       
W_60 = np.load('D:/ULTRASIP_Wmatrix_60.npy')       
W_75 = np.load('D:/ULTRASIP_Wmatrix_75.npy')       
W_90 = np.load('D:/ULTRASIP_Wmatrix_90.npy')       

cond_Wnone = cond_num(Wnone)
cond_W_15 = cond_num(W_15)
cond_W_30 = cond_num(W_30)
cond_W_45 = cond_num(W_45)
cond_W_60 = cond_num(W_60)
cond_W_75 = cond_num(W_75)
cond_W_90 = cond_num(W_90)

condition_numbers = [cond_Wnone,
cond_W_75,
cond_W_60,
cond_W_45,
cond_W_30,
cond_W_15]

px=200
py=100

pixel_values = []

for kappa in condition_numbers:
    
    kappa= kappa[1644:2000,1644:2000] 

    plot_condW(kappa,px,py)
    value = kappa[px,py]
    print(value)
    pixel_values = np.append(pixel_values,value)

plt.figure()
plt.plot(pixel_values,marker='o')
plt.xticks([0,1,2,3,4,5],[1,3,4,5,6,7])
plt.xlabel('Case #')
plt.ylabel('Pixel Value')
plt.title(f'Pixel {px,py}')









