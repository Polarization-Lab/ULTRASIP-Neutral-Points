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
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib as mpl
import h5py
import os
import glob

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

#------------------------Load Observation---------------------------------------
#Load observations 
#Set Date of Measurements 
date = '2025_10_22'

#Datapath
basepath = 'D:/Data'
#basepath = 'C:/Users/ULTRASIP_1/OneDrive/Desktop'
folderdate = os.path.join(basepath,date)
files = glob.glob(f'{folderdate}/*.h5')
f = h5py.File(files[7],'r+')
np_est = f['Neutral Point Estimation']

# Convert attributes to variables
for name, value in np_est.attrs.items():
    
    # make valid variable name
    var_name = name.replace(" ", "_").replace("-", "_").lower()
    
    # assign variable
    globals()[var_name] = value
    
    print(f"{var_name} = {value}")
    
aq = f[f'Aquistion_{aquisition_number}']

I = aq["UV Image Data/I_corrected"][:]
Q = aq["UV Image Data/Q_corrected"][:]
U = aq["UV Image Data/U_corrected"][:]

saz = aq['UV Image Data/sun_az'][()]
sza = aq['UV Image Data/sun_zen'][()]



vza = aq["UV Image Data/view_zen"][:]
vaz = aq["UV Image Data/view_az"][:]

vza = vza[:,0]
vaz = vaz[0,:]

q = Q/I
u=U/I

dolp = np.sqrt((q**2)+(u**2))*100
    
aolp = 0.5*np.arctan2(U,Q)
aolp = np.mod(np.degrees(aolp),180)



#-------------------------------------------------------------------------------

plt.figure(figsize=(16, 8))
plt.imshow(np.log(dolp),cmap='Blues_r',interpolation='None',vmin=-1,vmax=1)
plt.colorbar()

plt.figure(figsize=(16, 8))
plt.imshow(aolp,cmap=cmo.phase,interpolation='None', vmin=0, vmax=180)

plt.figure(figsize=(16, 8))
plt.imshow(I,cmap='gray',interpolation='None')
plt.title('I')
plt.colorbar()

plt.figure(figsize=(16, 8))
plt.imshow(Q/I,cmap=colmap,interpolation='None',vmin=-0.03,vmax=0.03)
plt.title('Q/I')
plt.colorbar()

plt.figure(figsize=(16, 8))
plt.imshow(U/I,cmap=colmap,interpolation='None',vmin=-0.03,vmax=0.03)
plt.title('U/I')
plt.colorbar()

fig, ax = plt.subplots(figsize=(8,8))

im = ax.imshow(q,
               cmap=colmap,
               interpolation='none',
               extent=[vaz.min(), vaz.max(), vza.max(), vza.min()],
               vmin=-0.02, vmax=0.02)

ax.set_xlabel('$\\gamma$ [$^\circ$]', fontsize=25)
ax.set_ylabel('$\\theta$ [$^\circ$]', fontsize=25)
ax.tick_params(axis='both', labelsize=23)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(0.5))   # or 0.25
ax.xaxis.set_major_locator(MultipleLocator(0.5))

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8,8))

im = ax.imshow(u,
               cmap=colmap,
               interpolation='none',
               extent=[vaz.min(), vaz.max(), vza.max(), vza.min()],
               vmin=-0.02, vmax=0.02)

ax.set_xlabel('$\\gamma$ [$^\circ$]', fontsize=25)
ax.set_ylabel('$\\theta$ [$^\circ$]', fontsize=25)
ax.tick_params(axis='both', labelsize=23)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(0.5))   # or 0.25
ax.xaxis.set_major_locator(MultipleLocator(0.5))

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
plt.show()



#------------------------------Cropping----------------------------------------

# Convert to slice objects
q_start, q_stop = map(int, q_cropped_region.split(':'))
u_start, u_stop = map(int, u_cropped_region.split(':'))

avgq = np.flip(np.average(q,axis=1))
avgu = np.average(u,axis=0)

avgq = avgq[q_start:q_stop]
avgu = avgu[u_start:u_stop]

vza_crop = vza[q_start:q_stop]
vaz_crop = vaz[u_start:u_stop]

# Calculate weights (inverse of standard deviation)
weights = (1 / np.std(vza_crop)) * np.ones_like(vza_crop)
               
# Add a constant (intercept) to the independent variable
avg_q_with_intercept = sm.add_constant(avgq)
               
# Weighted least squares regression
model = sm.WLS(vza_crop, avg_q_with_intercept, weights=weights)
qresults = model.fit()
               
# Get the fitted values and residuals
qfit_line = qresults.fittedvalues
qresiduals = vza_crop - qfit_line

qslope = qresults.params[1]
qint = qresults.params[0]
qint_stderror = qresults.bse[0]*3600

# Calculate weights (inverse of standard deviation)
weights = (1 / np.std(vaz_crop)) * np.ones_like(vaz_crop)
               
# Add a constant (intercept) to the independent variable
avg_u_with_intercept = sm.add_constant(avgu)
               
# Weighted least squares regression
model = sm.WLS(vaz_crop, avg_u_with_intercept, weights=weights)
uresults = model.fit()
               
# Get the fitted values and residuals
ufit_line = uresults.fittedvalues
residuals = vaz_crop - ufit_line

uslope = uresults.params[1]
uint = uresults.params[0]
uint_stderror = uresults.bse[0]*3600


# ---- Figure 1: Q vs Zenith ----
plt.figure(figsize=(12, 8))
plt.scatter(avgq, vza_crop, color='green')
plt.plot(avgq, qfit_line, color='gold', label='Weighted fitted line', linewidth=5)
plt.axvline(x=0, lw=5, color='red', zorder=0)

plt.text(-0.02, 38.5,
          f'$\\theta_s$: {sza:.2f}$^\circ$\nIntercept: {qint:.2f}$^\circ$ \n $SE_{{\\theta}}$: {qint_stderror:.2f} arcsec',
          fontsize=25,
          bbox=dict(facecolor='lightgray', alpha=1))

# Add labels and legend
plt.ylabel(r'$\theta$ [$\circ$]',fontsize=20)
plt.xlabel(r'$\bar{r_{Q}}$', fontsize = 20)
plt.xlim(-0.025, 0.025)
#plt.ylim([50.5, 54.5])
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.gca().invert_yaxis() 
#plt.title('Weighted Linear Regression with Fit Error')
plt.grid(True)

ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(0.5))   # or 0.25

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#plt.legend(fontsize=20,loc='upper left')
plt.show()


# ---- Figure 2: U vs Azimuth ----
plt.figure(figsize=(12, 8))
plt.scatter(avgu, vaz_crop, color='green')
plt.plot(avgu, ufit_line, color='gold', label='Weighted fitted line', linewidth=5)
plt.axvline(x=0, lw=5, color='red', zorder=0)
plt.axhline(y=saz,lw=5,color='orange')

plt.text(0.007, 10,
          f'$\\gamma_s$: {saz:.2f}$^\circ$ \nIntercept: {uint:.2f}$^\circ$ \n $SE_{{\\gamma}}$: {uint_stderror:.2f} arcsec',
          fontsize=25,
          bbox=dict(facecolor='lightgray', alpha=1))

# Add labels and legend
plt.ylabel(r'$\gamma$ [$\circ$]',fontsize=20)
plt.xlabel(r'$\bar{c_{U}}$', fontsize = 20)
plt.xlim(-0.025, 0.025)
#plt.ylim([50.5, 54.5])
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
#plt.title('Weighted Linear Regression with Fit Error')
plt.grid(True)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(0.5))   # or 0.25

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#plt.legend(fontsize=20,loc='upper left')
plt.show()


# ---------------- Q image ----------------
fig, ax = plt.subplots(figsize=(16, 10))

im = ax.imshow(q,
               cmap=colmap,
               interpolation='none',
               extent=[vaz.min(), vaz.max(), vza.max(), vza.min()],
               vmin=-0.02, vmax=0.02)

# ROI box
roi = Rectangle((vaz_crop.min(), vza_crop.min()),
                vaz_crop.max() - vaz_crop.min(),
                vza_crop.max() - vza_crop.min(),
                linewidth=3,
                edgecolor='white',
                facecolor='none')

ax.add_patch(roi)

ax.set_xlabel('$\gamma$ [$^\circ$]', fontsize=25)
ax.set_ylabel('$\\theta$ [$^\circ$]', fontsize=25)
ax.tick_params(axis='both', labelsize=23)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(0.5))   # or 0.25
ax.xaxis.set_major_locator(MultipleLocator(0.5))

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
plt.show()


# ---------------- U image ----------------
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(u,
               cmap=colmap,
               interpolation='none',
               extent=[vaz.min(), vaz.max(), vza.max(), vza.min()],
               vmin=-0.02, vmax=0.02)

# ROI box
roi = Rectangle((vaz_crop.min(), vza_crop.min()),
                vaz_crop.max() - vaz_crop.min(),
                vza_crop.max() - vza_crop.min(),
                linewidth=3,
                edgecolor='white',
                facecolor='none')

ax.add_patch(roi)

ax.set_xlabel('$\gamma$ [$^\circ$]', fontsize=25)
ax.set_ylabel('$\\theta$ [$^\circ$]', fontsize=25)
ax.tick_params(axis='both', labelsize=23)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(0.5))   # or 0.25
ax.xaxis.set_major_locator(MultipleLocator(0.5))

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
plt.show()

# ---------------- AoLP image ----------------
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(aolp,
               cmap=cmo.phase,
               interpolation='none',
               extent=[vaz.min(), vaz.max(), vza.max(), vza.min()],
               vmin=0, vmax=180)

# ROI box
roi = Rectangle((vaz_crop.min(), vza_crop.min()),
                vaz_crop.max() - vaz_crop.min(),
                vza_crop.max() - vza_crop.min(),
                linewidth=3,
                edgecolor='white',
                facecolor='none')

ax.add_patch(roi)

ax.scatter(uint, qint,
           s=400,
           color='red',
           edgecolor='black',
           linewidth=2,
           zorder=10)

ax.set_xlabel('$\gamma$ [$^\circ$]', fontsize=25)
ax.set_ylabel('$\\theta$ [$^\circ$]', fontsize=25)
ax.tick_params(axis='both', labelsize=23)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(0.5))   # or 0.25
ax.xaxis.set_major_locator(MultipleLocator(0.5))

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
plt.show()

# ---------------- DoLP image ----------------
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(np.log(dolp),
               cmap='Blues_r',
               interpolation='none',
               extent=[vaz.min(), vaz.max(), vza.max(), vza.min()],
               vmin=-0.2, vmax=1)

cbar = fig.colorbar(im, ax=ax,shrink=1,pad=0.01)
cbar.ax.tick_params(labelsize=20)

# ROI box
roi = Rectangle((vaz_crop.min(), vza_crop.min()),
                vaz_crop.max() - vaz_crop.min(),
                vza_crop.max() - vza_crop.min(),
                linewidth=3,
                edgecolor='white',
                facecolor='none')

ax.add_patch(roi)

ax.scatter(uint, qint,
           s=400,
           color='red',
           edgecolor='black',
           linewidth=2,
           zorder=10)
ax.set_xlabel('$\gamma$ [$^\circ$]', fontsize=25)
ax.set_ylabel('$\\theta$ [$^\circ$]', fontsize=25)
ax.tick_params(axis='both', labelsize=23)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(0.5))   # or 0.25
ax.xaxis.set_major_locator(MultipleLocator(0.5))

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(8,1))

norm = mpl.colors.Normalize(vmin=-0.02, vmax=0.02)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=colmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')

ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()