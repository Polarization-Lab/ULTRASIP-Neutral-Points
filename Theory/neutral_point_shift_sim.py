# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:14:47 2024

@author: ULTRASIP_1
"""

import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap


# Define W-matrix of ULTRASIP 
W_ultrasip = 0.5 * np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1]])

# Define input Stokes vector as a function of DoLP, AoLP, I
I = 1
# image dimensions
xsize = 2848
ysize = 2848

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


# make some empty placeholder arrays to fill later
aolp = np.zeros((xsize, ysize))
dolp = aolp.copy()

for xx in range(0, xsize, 1):  # for loop over x pixel index
    for yy in range(0, ysize, 1):  # for loop over y pixel index
        xidx = (xx - (xsize / 2)) / xsize
        yidx = (yy - (ysize / 2)) / ysize
        aolp[xx, yy] = np.arctan2(yidx, xidx)
        dolp[xx, yy] = np.sqrt((xidx**2) + (yidx**2))  # define DoLP with radial distance from singularity    

I = np.ones((xsize, ysize))
Q = I * dolp * np.cos(2 * aolp)
U = I * dolp * np.sin(2 * aolp)

plt.figure()
plt.imshow(Q/I,cmap=colmap,interpolation='None')
plt.title('Q/I')
plt.colorbar()

plt.figure()
plt.imshow(U/I,cmap=colmap,interpolation='None')
plt.title('U/I')
plt.colorbar()

aolp = np.mod(np.degrees(aolp), 180)

# Initialize arrays for vertical field
aolp_vert = np.zeros((xsize, ysize))
dolp_vert = aolp_vert.copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
I_vert = np.zeros((ysize, xsize))

# Initialize the plots
im1 = ax1.imshow(np.zeros((xsize, ysize)), cmap='GnBu_r', interpolation='None', vmin=0, vmax=70)
ax1.set_title('Degree of Linear Polarization (DoLP) [%]')
ax1.axis('off')
cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.8)

im2 = ax2.imshow(np.zeros((xsize, ysize)), cmap=cmo.phase, interpolation='None', vmin=0, vmax=180)
ax2.set_title('Angle of Linear Polarization (AoLP) [deg]')
ax2.axis('off')
cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.8)

def update(dolp_value):
    
    # Update dolp_vert with the current DoLP value
    for xx in range(0,xsize,1): # for loop over x pixel index
        for yy in range(0,ysize,1): # for loop over y pixel index
            xidx = (xx-(xsize/2))/xsize
            yidx = (yy-(ysize/2))/ysize
            aolp_vert[xx,yy] = np.pi/2
            dolp_vert[xx,yy] = dolp_value

    
    Q_vert = I * dolp_vert * np.cos(2 * aolp_vert)
    U_vert = I * dolp_vert * np.sin(2 * aolp_vert)
    
    Q_multi = Q[:] + Q_vert[:]
    U_multi = U[:] + U_vert[:]
    
    aolp_multi = np.mod(np.degrees(0.5 * np.arctan2(U_multi, Q_multi)), 180)
    dolp_multi = np.sqrt(Q_multi**2 + U_multi**2)
    
    # Update the plots
    # Update the data in the plots
    im1.set_data(dolp_multi * 100)
    im2.set_data(aolp_multi)
    fig.suptitle(f'Vertical Field DoLP: {dolp_value * 100:.1f}[%]')

# Create animation
dolp_values = [1] #np.arange(0, 1, 0.1)
ani = FuncAnimation(fig, update, frames=dolp_values, repeat=False)

# Save animation
ani.save("dolp_animation.mp4", writer='ffmpeg', fps=1)
plt.show()
plt.close('all')
