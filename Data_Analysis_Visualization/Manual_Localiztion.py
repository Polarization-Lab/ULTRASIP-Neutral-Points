"""
Created on Mon Mar  9 14:46:48 2026

@author: deleo

Neutral point localization simulation
"""

import numpy as np 
import matplotlib.pyplot as plt
import cmocean.cm as cmo 
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter
import statsmodels.api as sm

import h5py
import os
import glob

# ------------------------ Custom colormap ------------------------
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

# ------------------------ Load Observation ------------------------
date = '2025_10_24'

basepath = 'D:/Data'
folderdate = os.path.join(basepath, date)

files = glob.glob(f'{folderdate}/*.h5')
f = h5py.File(files[13], 'r+')
print(f)

aqnum = 6

# date = '2025_07_10'

# basepath = 'D:/Data'
# folderdate = os.path.join(basepath, date)

# files = glob.glob(f'{folderdate}/*.h5')
# f = h5py.File(files[5], 'r+')
# print(f)

# aqnum = 9
aq = f[f'Aquistion_{aqnum}']

timestamp = aq.attrs['Timestamp UTC']
print(timestamp)

I = aq["UV Image Data/I_corrected"][:]
Q = aq["UV Image Data/Q_corrected"][:]
U = aq["UV Image Data/U_corrected"][:]

saz = aq['UV Image Data/sun_az_corrected'][()]
sza = aq['UV Image Data/sun_zen_corrected'][()]

vza = aq["UV Image Data/view_zen_corrected"][:]
vaz = aq["UV Image Data/view_az_corrected"][:]

# Convert to 1D axes
vza = vza[:,0]
vaz = vaz[0,:]

q = Q/I
u=U/I

dolp = np.sqrt((q**2)+(u**2))*100
    
aolp = 0.5*np.arctan2(U,Q)
aolp = np.mod(np.degrees(aolp),180)

q_start, q_stop = [0,2500]
u_start, u_stop = [0,2500]


avgq = np.average(q,axis=1)
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
print(qresults.summary())
               
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
print(uresults.summary())

               
# Get the fitted values and residuals
ufit_line = uresults.fittedvalues
residuals = vaz_crop - ufit_line

uslope = uresults.params[1]
uint = uresults.params[0]
uint_stderror = uresults.bse[0]*3600

plt.figure()
plt.imshow(I,cmap='gray')
plt.colorbar()
plt.show()


# ---- Figure 1: Q vs Zenith ----
plt.figure(figsize=(12, 8))
plt.scatter(avgq, vza_crop, color='green')
plt.plot(avgq, qfit_line, color='gold', label='Weighted fitted line', linewidth=5)
plt.axvline(x=0, lw=5, color='red', zorder=0)

# plt.text(-0.02, 38.5,
#           f'$\\theta_s$: {sza:.2f}$^\circ$\nIntercept: {qint:.2f}$^\circ$ \n $SE_{{\\theta}}$: {qint_stderror:.2f} arcsec',
#           fontsize=25,
#           bbox=dict(facecolor='lightgray', alpha=1))

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

# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()

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

# plt.text(0.007, 10,
#           f'$\\gamma_s$: {saz:.2f}$^\circ$ \nIntercept: {uint:.2f}$^\circ$ \n $SE_{{\\gamma}}$: {uint_stderror:.2f} arcsec',
#           fontsize=25,
#           bbox=dict(facecolor='lightgray', alpha=1))

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
               extent=[vaz.min(), vaz.max(), vza.min(), vza.max()],
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

# ------------------------ Plot Q/I ------------------------
v = np.nanmax(np.abs(q))
norm = TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

plt.figure(figsize=(8,6))
plt.imshow(q, cmap='coolwarm', norm=norm, interpolation='none')
plt.title('Q/I (centered at 0)')
plt.colorbar(label='Q/I')
plt.show()

# ------------------------ Q vs U ------------------------
plt.figure(figsize=(8,8))
plt.scatter(q, u, s=5, alpha=0.5)

plt.xlabel('Q/I', fontsize=16)
plt.ylabel('U/I', fontsize=16)
plt.title('Q/I vs U/I')

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')

plt.grid()
plt.tight_layout()
plt.show()

# ------------------------ Neutral Point Detection ------------------------

# Normalize intensity
I_norm = I / np.nanmax(I)

# Mask: polarization + intensity constraints
mask = (
    (np.abs(q) <= 0.01) &
    (np.abs(u) <= 0.01) &
    (I_norm > 0.1) &
    (I_norm < 0.9)
)

# Compute distance to (0,0)
r = np.sqrt(q**2 + u**2)

# Smooth (reduces pixel noise influence)
r = gaussian_filter(r, sigma=2)

# Apply mask
r_masked = np.where(mask, r, np.inf)

# Safety check
if not np.any(mask):
    raise ValueError("No valid pixels after masking — relax thresholds")

# Find minimum
idx = np.unravel_index(np.argmin(r_masked), r.shape)
row_np, col_np = idx

# Convert to angles
vza_np = vza[row_np]
vaz_np = vaz[col_np]

print("Neutral Point Pixel:", row_np, col_np)
print(f"Neutral Point Angles → VZA: {vza_np:.2f}°, VAZ: {vaz_np:.2f}°")
print("sun az",saz)
print("sun zen",sza)

# Average DoLP over the region that was not masked 
avg_dolp = np.nanmean(dolp[mask])

print(f"Average DoLP in masked region: {avg_dolp}")
# Average DoLP over the region that was masked OUT
avg_dolp_outside = np.nanmean(dolp[~mask])

print(f"Average DoLP outside masked region: {avg_dolp_outside}")

# ------------------------ Visualize mask ------------------------
plt.figure(figsize=(8,6))
plt.imshow(mask, cmap='gray', interpolation='none')
plt.title('Valid NP Search Region')
plt.colorbar()
plt.show()

# ------------------------ Visualize NP ------------------------
plt.figure(figsize=(8,6))
im=plt.imshow(q, cmap=colmap, interpolation='none',vmin = -0.02,vmax=0.02)

plt.scatter(col_np, row_np, color='red', s=100, label='Neutral Point')

plt.title('Neutral Point on Q/I')
plt.colorbar(im,label='Q/I')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
im=plt.imshow(u, cmap=colmap, interpolation='none',vmin = -0.02,vmax=0.02)

plt.scatter(col_np, row_np, color='red', s=100, label='Neutral Point')

plt.title('Neutral Point on U/I')
plt.colorbar(im,label='U/I')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
im=plt.imshow(aolp, cmap=cmo.phase, interpolation='none',vmin =0,vmax=180)

plt.scatter(col_np, row_np, color='red', s=100, label='Neutral Point')

plt.title('Neutral Point on AoLP')
plt.colorbar(im,label='AoLP')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
im=plt.imshow(np.log(dolp), cmap='Blues_r', interpolation='none',vmin =-2,vmax=0)

plt.scatter(col_np, row_np, color='red', s=100, label='Neutral Point')

plt.title('Neutral Point on DoLP')
plt.colorbar(im,label='log(DoLP)')
plt.legend()
plt.show()

# save = input("Save neutral point estimation to file? (Yes/No): ")
# if save.lower() in ['yes', 'y']:
#     if 'Manual Neutral Point Estimation' not in f:
#         np_est = f.create_group("Manual Neutral Point Estimation")
#     else: 
#         del f['Manual Neutral Point Estimation']
#         np_est = f.create_group("Manual Neutral Point Estimation")

#     np_est.create_dataset('Manual Estimation NP Location (zen,az) [deg]', data = np.array([vza_np, vaz_np]))
#     np_est.create_dataset('Sun Location (zen,az) [deg]', data = np.array([sza, saz]))
#     np_est.attrs['Aquisition Number'] = aqnum
#     np_est.attrs['Time Stamp'] = timestamp
#     print("Manual Neutral point estimation saved.")
#     f['Measurement_Metadata'].attrs['Processed Level'] = 'Level 3'

# ---------------- Q image ----------------
fig, ax = plt.subplots(figsize=(16, 10))

im = ax.imshow(q,
               cmap=colmap,
               interpolation='none',
               extent=[vaz.min(), vaz.max(), vza.max(), vza.min()],
               vmin=-0.02, vmax=0.02)


ax.scatter(vaz[col_np], vza[row_np],
           s=400,
           color='red',
           edgecolor='black',
           linewidth=2,
           zorder=10)

ax.set_xlabel('$\gamma$ [$^\circ$]', fontsize=25)
ax.set_ylabel('$\\theta$ [$^\circ$]', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(1))   # or 0.25
ax.xaxis.set_major_locator(MultipleLocator(1))

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
plt.show()


# ---------------- Q image ----------------
fig, ax = plt.subplots(figsize=(16, 10))

im = ax.imshow(u,
               cmap=colmap,
               interpolation='none',
               extent=[vaz.min(), vaz.max(), vza.max(), vza.min()],
               vmin=-0.02, vmax=0.02)


ax.scatter(vaz[col_np], vza[row_np],
           s=400,
           color='red',
           edgecolor='black',
           linewidth=2,
           zorder=10)

ax.set_xlabel('$\gamma$ [$^\circ$]', fontsize=25)
ax.set_ylabel('$\\theta$ [$^\circ$]', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(1))   # or 0.25
ax.xaxis.set_major_locator(MultipleLocator(1))

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


ax.scatter(vaz[col_np], vza[row_np],
           s=400,
           color='red',
           edgecolor='black',
           linewidth=2,
           zorder=10)

ax.set_xlabel('$\gamma$ [$^\circ$]', fontsize=25)
ax.set_ylabel('$\\theta$ [$^\circ$]', fontsize=25)
ax.tick_params(axis='both', labelsize=23)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(1))   # or 0.25
ax.xaxis.set_major_locator(MultipleLocator(1))

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
               vmin = -0.2, vmax = 1.00
               )

cbar = fig.colorbar(im, ax=ax,shrink=1,pad=0.01)
cbar.ax.tick_params(labelsize=20)


ax.scatter(vaz[col_np], vza[row_np],
           s=400,
           color='red',
           edgecolor='black',
           linewidth=2,
           zorder=10)

ax.set_xlabel('$\gamma$ [$^\circ$]', fontsize=25)
ax.set_ylabel('$\\theta$ [$^\circ$]', fontsize=25)
ax.tick_params(axis='both', labelsize=23)
plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(1))   # or 0.25
ax.xaxis.set_major_locator(MultipleLocator(1))

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.tight_layout()
plt.show()




# ---- Figure 1: Q vs Zenith ----
plt.figure(figsize=(12, 8))
plt.scatter(avgq, vza_crop, color='green')
plt.plot(avgq, qfit_line, color='gold', label='Weighted fitted line', linewidth=5)
plt.axvline(x=0, lw=5, color='red', zorder=0)


# Add labels and legend
plt.ylabel(r'$\theta$ [$\circ$]',fontsize=30)
plt.xlabel(r'$\bar{r_{Q}}$', fontsize = 30)
plt.xlim(-0.025, 0.025)
#plt.ylim([50.5, 54.5])
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
#plt.title('Weighted Linear Regression with Fit Error')
plt.grid(True)

ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(1))   # or 0.25

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#plt.legend(fontsize=20,loc='upper left')
plt.show()


# ---- Figure 2: U vs Azimuth ----
plt.figure(figsize=(12, 8))
plt.scatter(avgu, vaz_crop, color='green')
plt.plot(avgu, ufit_line, color='gold', label='Weighted fitted line', linewidth=5)
plt.axvline(x=0, lw=5, color='red', zorder=0)
#plt.axhline(y=saz,lw=5,color='orange')


# Add labels and legend
plt.ylabel(r'$\gamma$ [$\circ$]',fontsize=30)
plt.xlabel(r'$\bar{c_{U}}$', fontsize = 30)
plt.xlim(-0.025, 0.025)
#plt.ylim([50.5, 54.5])
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
#plt.title('Weighted Linear Regression with Fit Error')
plt.grid(True)
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(1))   # or 0.25

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#plt.legend(fontsize=20,loc='upper left')
plt.show()