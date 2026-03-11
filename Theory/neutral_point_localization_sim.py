# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:46:48 2026

@author: deleo

Neutral point localization simulation
"""

import numpy as np 
import matplotlib.pyplot as plt
import cmocean.cm as cmo 
from matplotlib.colors import ListedColormap
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib as mpl



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

# Idealized W-matrix
W_ultrasip = 0.5 * np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1]])

# Define input Stokes vector as a function of DoLP, AoLP, I
I = 1
# image dimensions
xsize = 2848
ysize = 2848

# make some empty placeholder arrays to fill later
aolp = np.zeros((xsize, ysize))
dolp = aolp.copy()

for xx in range(0, xsize, 1):  # for loop over x pixel index
    for yy in range(0, ysize, 1):  # for loop over y pixel index
        xidx = (xx - (xsize/2)) / xsize
        yidx = (yy - (ysize/2)) / ysize
        aolp[xx, yy] = -0.5*np.arctan2(yidx, -xidx)
        dolp[xx, yy] = np.sqrt((xidx**2) + (yidx**2))  # define DoLP with radial distance from singularity    

I = np.ones((xsize, ysize))
Q = I * dolp * np.cos(2*aolp)
U = I * dolp * np.sin(2*aolp)

plt.figure()
plt.imshow(dolp*100,cmap='Blues_r',interpolation='None')
plt.title('DoLP [%]')
plt.colorbar()

plt.figure()
plt.imshow(np.mod(np.degrees(aolp),180),cmap=cmo.phase,interpolation='None', vmin=0, vmax=180)
plt.title('AoLP [$\circ$]')
plt.colorbar()

plt.figure()
plt.imshow(I,cmap='gray',interpolation='None')
plt.title('I')
plt.colorbar()

plt.figure()
plt.imshow(Q/I,cmap=colmap,interpolation='None')
plt.title('Q/I')
plt.colorbar()

plt.figure()
plt.imshow(U/I,cmap=colmap,interpolation='None')
plt.title('U/I')
plt.colorbar()

q = Q/I
u=U/I

avgq = np.average(q,axis=1)
avgu = np.average(u,axis=0)

# vza=np.arange(0,len(avgq))
# vaz=np.arange(0,len(avgu))

step = 5.78/2847
vza=np.arange(59.22,65.00,step)
vaz=np.arange(-75.00,-69.22,step)


# Calculate weights (inverse of standard deviation)
weights = (1 / np.std(vza)) * np.ones_like(vza)
               
# Add a constant (intercept) to the independent variable
avg_q_with_intercept = sm.add_constant(avgq)
               
# Weighted least squares regression
model = sm.WLS(vza, avg_q_with_intercept, weights=weights)
qresults = model.fit()
               
# Get the fitted values and residuals
qfit_line = qresults.fittedvalues
qresiduals = vza - qfit_line

qslope = qresults.params[1]
qint = qresults.params[0]
qint_stderror = qresults.bse[0]*3600

# Calculate weights (inverse of standard deviation)
weights = (1 / np.std(vaz)) * np.ones_like(vaz)
               
# Add a constant (intercept) to the independent variable
avg_u_with_intercept = sm.add_constant(avgu)
               
# Weighted least squares regression
model = sm.WLS(vaz, avg_u_with_intercept, weights=weights)
uresults = model.fit()
               
# Get the fitted values and residuals
ufit_line = uresults.fittedvalues
residuals = vaz - ufit_line

uslope = uresults.params[1]
uint = uresults.params[0]
uint_stderror = uresults.bse[0]*3600


# ---- Figure 1: Q vs Zenith ----
plt.figure(figsize=(6,5))

plt.scatter(avgq, vza, color='green')
plt.plot(avgq, qfit_line, color='gold', label='Weighted fitted line', linewidth=3)
plt.axvline(x=0, lw=5, color='red', zorder=0)

plt.xlabel(r'$\bar{r}_{Q}$', fontsize=18)
plt.ylabel('Zenith [$^\circ$]', fontsize=18)

plt.text(-0.65, 60,
         f'Intercept: {qint:.4f}$^\circ$ \n $\sigma_{{zen}}$: {qint_stderror:.4e} arcsec',
         fontsize=16,
         bbox=dict(facecolor='lightgray', alpha=1))

plt.xlim(-0.7, 0.7)
plt.ylim(59, 65.2)
plt.gca().invert_yaxis()

plt.tick_params(axis='both', labelsize=15)
plt.grid()
plt.tight_layout()
plt.show()


# ---- Figure 2: U vs Azimuth ----
plt.figure(figsize=(6,5))

plt.scatter(avgu, vaz, color='green')
plt.plot(avgu, ufit_line, color='gold', label='Weighted fitted line', linewidth=3)
plt.axvline(x=0, lw=5, color='red', zorder=0)

plt.xlabel(r'$\bar{c}_{U}$', fontsize=18)
plt.ylabel('Azimuth [$^\circ$]', fontsize=18)

plt.text(-0.1, -70,
         f'Intercept: {uint:.4f}$^\circ$ \n $\sigma_{{az}}$: {uint_stderror:.4e} arcsec',
         fontsize=16,
         bbox=dict(facecolor='lightgray', alpha=1))

plt.xlim(-0.7, 0.7)
plt.ylim(-75.2, -69)

plt.tick_params(axis='both', labelsize=15)
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))

im = plt.imshow(q,
                cmap=colmap,
                interpolation='none',
                extent=[vaz.min(), vaz.max(), vza.max(), vza.min()])

plt.xlabel('Azimuth [$^\circ$]', fontsize=16)
plt.ylabel('Zenith [$^\circ$]', fontsize=16)
plt.tick_params(axis='both', labelsize=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))

im = plt.imshow(u,
                cmap=colmap,
                interpolation='none',
                extent=[vaz.min(), vaz.max(), vza.max(), vza.min()])

plt.xlabel('Azimuth [$^\circ$]', fontsize=16)
plt.ylabel('Zenith [$^\circ$]', fontsize=16)
plt.tick_params(axis='both', labelsize=15)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(6,1))

cbar = plt.colorbar(im,
                    orientation='horizontal',
                    fraction=0.8,
                    pad=0.2)

cbar.set_label('Normalized Stokes Parameter')

plt.axis('off')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8,1))

norm = mpl.colors.Normalize(vmin=np.min(q), vmax=np.max(q))
sm = mpl.cm.ScalarMappable(norm=norm, cmap=colmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')

ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()