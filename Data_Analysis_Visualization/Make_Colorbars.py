# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:58:14 2026

@author: ULTRASIP_1
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cmocean.cm as cmo
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

# Create figure
fig, ax = plt.subplots(figsize=(8, 1))

# Normalization
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sm = mpl.cm.ScalarMappable(cmap='Blues_r', norm=norm)
sm.set_array([])

# Horizontal colorbar
cbar = plt.colorbar(
    sm,
    cax=ax,
    orientation='horizontal'
)

# Set ONLY desired ticks
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(['-1', '0', '1'])

cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.show()

# Create figure
fig, ax = plt.subplots(figsize=(8, 1))

# Normalization
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sm = mpl.cm.ScalarMappable(cmap=colmap, norm=norm)
sm.set_array([])

# Horizontal colorbar
cbar = plt.colorbar(
    sm,
    cax=ax,
    orientation = 'horizontal'
)

# Set ONLY desired ticks
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(['$I_\parallel > I_\perp$','$I_\parallel = I_\perp$','$I_\parallel < I_\perp$'])

cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.show()

# Create figure
fig, ax = plt.subplots(figsize=(8, 1))

# Normalization
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sm = mpl.cm.ScalarMappable(cmap='Blues_r', norm=norm)
sm.set_array([])

# Horizontal colorbar
cbar = plt.colorbar(
    sm,
    cax=ax,
    orientation='horizontal'
)

# Set ONLY desired ticks
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(['-1', '0', '1'])

cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.show()

# Create figure
fig, ax = plt.subplots(figsize=(8, 1))

# Normalization
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sm = mpl.cm.ScalarMappable(cmap=colmap, norm=norm)
sm.set_array([])

# Horizontal colorbar
cbar = plt.colorbar(
    sm,
    cax=ax,
    orientation='horizontal'
)

# Set ONLY desired ticks
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels([
    r'$I_{\nearrow} > I_{\searrow}$',
    r'$I_{\nearrow} = I_{\searrow}$',
    r'$I_{\nearrow} < I_{\searrow}$'
])

cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.show()



# Create figure
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection='polar')

# Hemisphere angular range (0 → π)
theta = np.linspace(0, np.pi, 600)
r = np.linspace(0, 1, 200)
Theta, R = np.meshgrid(theta, r)

# AoLP definition (mod 180°)
aolp = np.mod(np.degrees(Theta)/2, 180)

# Plot hemisphere
ax.pcolormesh(Theta, R, aolp, cmap=cmo.phase, shading='auto')

# Make it look like a sky dome
ax.set_theta_zero_location("N")     # 0° at top
ax.set_theta_direction(-1)          # clockwise

# Remove everything
ax.set_axis_off()

plt.tight_layout()
plt.savefig('AoLP_hemisphere_colorwheel.png', dpi=300, transparent=True)
plt.show()
