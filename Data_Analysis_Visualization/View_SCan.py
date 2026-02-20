# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 13:16:33 2026

@author: ULTRASIP_1
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import cmocean.cm as cmo
import glob
import os
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap


#Custom colormap for Q and U
#Blue to Red Color scale for S1 and S2
colmap = np.zeros((255,3));
# Red
colmap[126:183,0]= np.linspace(0,1,57);
colmap[183:255,0]= 1;
# Green
colmap[0:96,1] = np.linspace(1,0,96);
colmap[158:255,1]= np.linspace(0,1,97);
# Blue
colmap[0:71,2] = 1;
colmap[71:128,2]= np.linspace(1,0,57);
colmap2 = colmap[128:,:]
colmap = ListedColormap(colmap)

# =========================
# Load observations
# =========================
date = '2025_10_22'
basepath = 'D:/Data'

folderdate = os.path.join(basepath, date)
file = glob.glob(f'{folderdate}/*_14_15*.h5')

f = h5py.File(file[0], 'r')

# =========================
# Create FIXED figure
# =========================
fig = plt.figure(figsize=(18, 8), dpi=100, constrained_layout=False)

# Expand left margin so ylabel is never clipped
ax1 = fig.add_axes([0.10, 0.12, 0.38, 0.78])   # left panel
ax2 = fig.add_axes([0.57, 0.12, 0.38, 0.78])   # right panel


for aqnum in range(0, 10):

    print("Processing acquisition:", aqnum)

    aq = f[f'Aquistion_{aqnum}']

    view_az = aq['UV Image Data/view_az'][:]
    view_zen = aq['UV Image Data/view_zen'][:]

    I = aq['UV Image Data/I_corrected'][:]
    Q = aq['UV Image Data/Q_corrected'][:]
    U = aq['UV Image Data/U_corrected'][:]

    q = Q / I
    u = U / I

    dolp = np.sqrt(q**2 + u**2) * 100
    aolp = 0.5 * np.arctan2(U, Q)
    aolp = np.mod(np.degrees(aolp), 180)
        
    if aqnum == 0:
        view_zen = 90 - view_zen 

    center_az = view_az[1424,1424]
    center_zen = view_zen[1424,1424]

    fig = plt.figure(figsize=(18, 8), dpi=100, constrained_layout=False)

    # Expand left margin so ylabel is never clipped
    ax1 = fig.add_axes([0.10, 0.12, 0.38, 0.78])   # left panel
    ax2 = fig.add_axes([0.57, 0.12, 0.38, 0.78])   # right panel

    im1 = ax1.imshow(
    np.log(dolp),
    cmap='Blues_r',
    vmin=-0.5,
    vmax=1,
    extent=[view_az.min(), view_az.max(),
            view_zen.max(), view_zen.min()],
    interpolation='none'
    )

    im2 = ax2.imshow(
                aolp,
                cmap=cmo.phase,
                vmin=0,
                vmax=180,
                extent=[view_az.min(), view_az.max(),
                        view_zen.max(), view_zen.min()],
                interpolation='none'
        )


    # LOCK LIMITS (never change)
    ax1.set_xlim(view_az.min(), view_az.max())
    ax1.set_ylim(view_zen.max(), view_zen.min())
    cbar = fig.colorbar(im1, ax=ax1, shrink = 0.8)
    cbar.ax.tick_params(labelsize=22)
    ax2.set_xlim(view_az.min(), view_az.max())
    ax2.set_ylim(view_zen.max(), view_zen.min())

    # Labels + titles (set once)
    ax1.set_title('log(Degree of Linear Polarization [%])', fontsize=26)
    ax1.set_xlabel('Azimuth [$\circ$]', fontsize=25)
    ax1.set_ylabel('Zenith [$\circ$]', fontsize=25, labelpad=20)

    ax2.set_title('Angle of Linear Polarization [$\circ$]', fontsize=26)
    ax2.set_xlabel('Azimuth [$\circ$]', fontsize=25)
    ax2.set_ylabel('Zenith [$\circ$]', fontsize=25, labelpad=20)


    # Center ticks only
    ax1.set_xticks([center_az])
    ax1.set_yticks([center_zen])
    ax1.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
    ax1.set_yticklabels([f'{center_zen:.1f}°'], fontsize=25)

    ax2.set_xticks([center_az])
    ax2.set_yticks([center_zen])

    ax2.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
    ax2.set_yticklabels([])
    
    fig = plt.figure(figsize=(18, 8), dpi=100, constrained_layout=False)

    # Expand left margin so ylabel is never clipped
    ax1 = fig.add_axes([0.10, 0.12, 0.38, 0.78])   # left panel
    ax2 = fig.add_axes([0.57, 0.12, 0.38, 0.78])   # right panel

    im1 = ax1.imshow(
    q,
    cmap=colmap,
    vmin=-0.03,
    vmax=0.03,
    extent=[view_az.min(), view_az.max(),
            view_zen.max(), view_zen.min()],
    interpolation='none'
    )

    im2 = ax2.imshow(
                u,
                cmap=colmap,
                vmin=-0.03,
                vmax=0.03,
                extent=[view_az.min(), view_az.max(),
                        view_zen.max(), view_zen.min()],
                interpolation='none'
        )


    # LOCK LIMITS (never change)
    ax1.set_xlim(view_az.min(), view_az.max())
    ax1.set_ylim(view_zen.max(), view_zen.min())
    ax2.set_xlim(view_az.min(), view_az.max())
    ax2.set_ylim(view_zen.max(), view_zen.min())
    # cbar = fig.colorbar(im1, ax=ax1, shrink = 0.8)
    # cbar.ax.tick_params(labelsize=22)
    cbar = fig.colorbar(im2, ax=ax2, shrink = 0.8)
    cbar.ax.tick_params(labelsize=22)


    # Labels + titles (set once)
    ax1.set_title('Q/I', fontsize=26)
    ax1.set_xlabel('Azimuth [$\circ$]', fontsize=25)
    ax1.set_ylabel('Zenith [$\circ$]', fontsize=25, labelpad=20)

    ax2.set_title('U/I', fontsize=26)
    ax2.set_xlabel('Azimuth [$\circ$]', fontsize=25)
    ax2.set_ylabel('Zenith [$\circ$]', fontsize=25, labelpad=20)


    # Center ticks only
    ax1.set_xticks([center_az])
    ax1.set_yticks([center_zen])
    ax1.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
    ax1.set_yticklabels([f'{center_zen:.1f}°'], fontsize=25)

    ax2.set_xticks([center_az])
    ax2.set_yticks([center_zen])

    ax2.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
    ax2.set_yticklabels([])