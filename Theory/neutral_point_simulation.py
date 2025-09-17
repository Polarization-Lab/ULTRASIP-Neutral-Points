# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:07:03 2024

@author: ULTRASIP_1
"""
# Import Libraries 
import numpy as np
import matplotlib.pyplot as plt
from colorwheel import colorwheel_imshow as cwimshow
import cmocean.cm as cmo

# Define W-matrix of ULTRASIP 
# The analyzer vectors (P0,P90,P45,P135) are the rows of the W-matrix (pg 230 of PL&OS)
W_ultrasip = 0.5 * np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1]])

# Define input Stokes vector as a function of DoLP, AoLP, I
I = 1
# image dimensions
xsize = 2848
ysize = 2848

# make some empty placeholder arrays to fill later
aolp = np.zeros((xsize, ysize))
dolp = aolp.copy()

for xx in range(0, xsize, 1): # for loop over x pixel index
    for yy in range(0, ysize, 1): # for loop over y pixel index
        xidx = (xx - (xsize / 2)) / xsize
        yidx = (yy - (ysize / 2)) / ysize
        aolp[xx, yy] = np.arctan2(yidx, xidx) 
        dolp[xx, yy] = 1*np.sqrt((xidx ** 2) + (yidx ** 2)) # define DoLP with radial distance from singularity    

I = np.ones((xsize, ysize))
Q = I * dolp * np.cos(2 * aolp)
U = I * dolp * np.sin(2 * aolp)

Stokes_in = np.array([I, Q, U])

aolp = np.mod(np.degrees(aolp), 180)

aolp_vert = np.zeros((xsize, ysize))
dolp_vert = aolp_vert.copy()

for xx in range(0, xsize, 1): # for loop over x pixel index
    for yy in range(1024, 1724, 1): # for loop over y pixel index
        xidx = (xx - (xsize / 2)) / xsize
        yidx = (yy - (ysize / 2)) / ysize
        aolp_vert[xx, yy] = np.pi / 2
        dolp_vert[xx, yy] = 0.15

I_vert = np.zeros((ysize, xsize))
Q_vert = I * dolp_vert * np.cos(2 * aolp_vert)
U_vert = I * dolp_vert * np.sin(2 * aolp_vert)

Stokes_vert = np.array([I_vert, Q_vert, U_vert])

# # w/multi
dist_from_center = np.sqrt(((np.arange(xsize) - xsize / 2)[:, None] / xsize)**2 + ((np.arange(ysize) - ysize / 2)[None, :] / ysize)**2)
weight_center = 1 - dist_from_center / np.max(dist_from_center)

# Q_multi = Q * weight_center + Q_vert * (1 - weight_center)
# U_multi = U * weight_center + U_vert * (1 - weight_center)

Q_multi = Q  + Q_vert 
U_multi = U  + U_vert 

aolp_multi = np.mod(np.degrees(0.5*np.arctan2(U_multi, Q_multi)), 180)
dolp_multi = np.sqrt(Q_multi ** 2 + U_multi ** 2)

center_x = int(xsize/2)
center_y = int(ysize/2)

#dolp_multi[center_x-150:center_x+150,center_y-150:center_y+150]=0

# Simulate a blue sky with a gradient centered on the sun
blue_sky = np.zeros((ysize, xsize, 3), dtype=np.uint8)
for xx in range(0, xsize, 1):
    for yy in range(0, ysize, 1):
        xidx = (xx - (xsize / 2)) / xsize
        yidx = (yy - (ysize / 2)) / ysize
        dist_to_center = np.sqrt(xidx**2 + yidx**2)
        blue_intensity = min(255, 255 * np.exp(-dist_to_center * 4))  # Create a linear gradient
        blue_intensity = np.clip(blue_intensity, 0, 255)
        blue_sky[yy, xx] = [blue_intensity, blue_intensity, 255]


#plots

plt.figure()
plt.imshow(blue_sky)
plt.axis('off')

# single scatter
sample_step = 200
Xs, Ys = np.meshgrid(np.arange(0, xsize, sample_step), np.arange(0, ysize, sample_step))
Us_vec = dolp[::sample_step, ::sample_step] * np.cos(np.radians(aolp[::sample_step, ::sample_step]))
Vs_vec = dolp[::sample_step, ::sample_step] * np.sin(np.radians(aolp[::sample_step, ::sample_step]))

Xvert, Yvert = np.meshgrid(np.arange(0, len(dolp), sample_step), np.arange(0, len(dolp), sample_step))
Uvert_vec = dolp_vert[::sample_step, ::sample_step] * np.cos(aolp_vert[::sample_step, ::sample_step])
Vvert_vec = dolp_vert[::sample_step, ::sample_step] * np.sin(aolp_vert[::sample_step, ::sample_step])

# Compute magnitude of vectors
mag_vert = np.sqrt(Uvert_vec**2 + Vvert_vec**2)

# Mask small/zero vectors
mask = mag_vert > 1e-6   # tolerance 

plt.figure(figsize=(10, 10))
plt.imshow(blue_sky)

# Lime (sky pattern)
plt.quiver(Xs, Ys, Us_vec, -Vs_vec, 
           color='lime', headwidth=0, angles='xy', 
           scale_units='xy', scale=0.0015, width=0.006)

# Hotpink (vertical scatter) — only where mask is true
plt.quiver(Xvert[mask], Yvert[mask], 
           Uvert_vec[mask], Vvert_vec[mask], 
           color='magenta', headwidth=0, angles='xy', 
           scale_units='xy', scale=0.0012, width=0.006)

plt.gca().invert_yaxis()
plt.axis('off')
plt.show()


# multi scatter
sample_step =200
X, Y = np.meshgrid(np.arange(0,xsize, sample_step), np.arange(0, ysize, sample_step))
U_vec = dolp_multi[::sample_step, ::sample_step] * np.cos(np.radians(aolp_multi[::sample_step, ::sample_step]))
V_vec = dolp_multi[::sample_step, ::sample_step] * np.sin(np.radians(aolp_multi[::sample_step, ::sample_step]))


# Compute magnitude of vectors
mag_multi = np.sqrt(U_vec**2 + V_vec**2)

# Mask small/zero vectors
maskm = mag_multi > 1e-4  # tolerance 

plt.figure(figsize=(10, 10))
plt.imshow(blue_sky)
plt.quiver(X[maskm], Y[maskm], U_vec[maskm], -V_vec[maskm], color='white', headwidth=0,angles='xy', scale_units='xy', scale=0.0015, width=0.006)
#streamplot = plt.streamplot(X, Y, U_vec, -V_vec, color='white', linewidth=0.5, density=2,arrowsize=0)

plt.gca().invert_yaxis()
plt.axis('off')
plt.show()


# plt.figure()
# plt.imshow(dolp*100,cmap='GnBu_r',interpolation= 'None',vmin=0, vmax=70)
# plt.colorbar()
# plt.title('Degree of Linear Polarization (DoLP) [%]')
# plt.axis('off')

# f=plt.figure(figsize=(15, 8))
# title = 'Angle of Linear Polarization (AoLP) [deg]'
# cwimshow(f,aolp,title)

# plt.figure()
# plt.imshow(dolp_multi*100,cmap='GnBu_r',interpolation= 'None',vmin=0, vmax=70)
# plt.colorbar()
# plt.title('Degree of Linear Polarization (DoLP) [%]')
# plt.axis('off')

# f=plt.figure(figsize=(15, 8))
# title = 'Angle of Linear Polarization (AoLP) [deg]'
# cwimshow(f,aolp_multi,title)


# #Create flux images from aolp_multi and dolp_multi
# I_o = np.zeros((ysize, xsize))
# Q_o = I * dolp_multi * np.cos(2 * np.radians(aolp_multi))
# U_o = I * dolp_multi * np.sin(2 * np.radians(aolp_multi))

# Stokes_o = np.array([I_o, Q_o, U_o]).reshape(3,xsize*ysize)

# P = np.linalg.pinv(W_ultrasip).T@Stokes_o
# P = P.reshape(4,xsize,ysize)

# P0 = P[0,:,:]
# P90 = P[1,:,:]
# P45 = P[2,:,:]
# P135 = P[3,:,:]

# plt.figure()
# plt.imshow(P0)

# aolp = 0.5*np.arctan2(U_o,Q_o)
# aolp = np.mod(np.degrees(aolp),180)
# dolp = np.sqrt(U_o**2 + Q_o**2)/I

# plt.figure()
# #plt.title('Angle of Linear Polarization (AoLP) [$\circ$]')
# plt.imshow(aolp,cmap=cmo.phase,interpolation= 'None')
# #cb = plt.colorbar()
# #cb.ax.tick_params(labelsize=10)
# plt.axis('off')

# f=plt.figure(figsize=(15, 8))
# title = 'AoLP Meas [deg]'
# cwimshow(f,aolp,title)

# plt.figure()
# plt.imshow(np.log(dolp),cmap='Blues_r',interpolation= 'None',vmin=-5, vmax=0)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=15)
# plt.title('Log of Degree of Linear Polarization (DoLP)')
# plt.axis('off')

# # Simulate Flux Measurements of a Zoomed in Picture of the Neutral Point
# Stokes_meas = (Stokes_in[:]+Stokes_vert[:]).reshape(3,xsize*ysize)

# #P= pinv(W)@S_meas

# P = np.linalg.pinv(W_ultrasip).T@Stokes_meas
# P = P.reshape(4,xsize,ysize)
# # P0=P[0,900:1100,1300:1600]
# # P90=P[1,900:1100,1300:1600]
# # P45=P[2,900:1100,1300:1600]
# # P135=P[3,900:1100,1300:1600]

# P0=P[0,:,:]
# P90=P[1,:,:]
# P45=P[2,:,:]
# P135=P[3,:,:]

# Q_meas = P0[:]-P90[:]
# U_meas = P45[:] - P135[:]

# dolp_meas = np.sqrt(Q_meas**2 + U_meas**2)
# aolp_meas = np.mod(np.degrees(0.5*np.arctan2(U_meas,Q_meas)),180)

# plt.figure()
# plt.imshow(dolp_meas*100,cmap='hot',interpolation= 'None',vmin=0, vmax=8)
# plt.colorbar()
# plt.title('DoLP Meas [%]')

# f=plt.figure(figsize=(15, 8))
# title = 'AoLP Meas [deg]'
# cwimshow(f,aolp_meas,title)

# #axis 1 is along columns, axis=0 is along rows
# avg0 = np.average(P0, axis=1)
# avg135 = np.average(P135, axis=0)
# avg90 = np.average(P90, axis=1)
# avg45 = np.average(P45, axis=0)
 
# # Finding the overlap points
# overlap_indices = np.where(np.isclose(avg45, avg135))[0]
# xoverlap = overlap_indices

# plt.figure()
# plt.scatter(range(0,len(avg45)),avg45,label='P[45]',color='blue')
# plt.scatter(range(0,len(avg135)),avg135,label='P[135]',color='orange')
# # Highlighting the overlap points
# plt.scatter(overlap_indices, avg45[overlap_indices], color='black',marker='x', label=f'Pixel{xoverlap}',s=100)
# # Drawing vertical lines at the overlap points
# for index in overlap_indices:
#     plt.axvline(x=index, color='black', linestyle='--', alpha=0.8)
# plt.ylabel("Digital Count")
# plt.xlabel("X-Dimension Pixel Index")
# plt.title("Average Flux Values")
# # plt.ylim(0, 1)
# plt.grid()
# plt.minorticks_on()
# plt.legend()
# plt.show()

# plt.figure()
# plt.scatter(range(0,len(avg90)),avg90,label='P[90]',color='green')
# plt.scatter(range(0,len(avg0)),avg0,label='P[0]',color='red')

# # Finding the overlap points
# yoverlap_indices = np.where(np.isclose(avg0, avg90,atol=0.0001))[0]
# yoverlap = yoverlap_indices
        
# # Highlighting the overlap points
# plt.scatter(yoverlap_indices, avg0[yoverlap_indices], color='black',marker='x', label=f'Pixel{yoverlap}',s=100)

# # Drawing vertical lines at the overlap points
# for index in yoverlap_indices:
#     plt.axvline(x=index, color='black', linestyle='--', alpha=0.8)

# plt.ylabel("Digital Count")
# plt.xlabel("Y-Dimension Pixel Index")
# plt.title("Average Flux Values")
# # plt.ylim(0, 1)
# plt.grid()
# plt.minorticks_on()
# plt.legend()
# plt.show()


