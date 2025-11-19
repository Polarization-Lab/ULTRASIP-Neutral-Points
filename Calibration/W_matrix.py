# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:12:35 2025

@author: ULTRASIP_1

Find W-matrix 
"""

#Import libraries 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import cmocean.cm as cmo
import glob
import h5py
import os
# Define W-matrix of ULTRASIP (rows = analyzer vectors P0, P90, P45, P135)
W = 0.5 * np.array([[1, 1, 0],[1, -1, 0],[1, 0, 1],[1, 0, -1]])
#W-matrix cal 
Stokes_ideal = np.array([[1,1,1,1],[1,-1,0,0],[0,0,1,-1]])
data = np.load('D:/NUC_0813.npz', allow_pickle=True)

Rij = data['arr1'].item()   # convert array-object → Python dict
Bij = data['arr2'].item()

#Correct image
def correct_img(Pij,Rij,Bij):
    R_avg = np.mean(Rij)
    B_avg = np.mean(Bij)
    Cij = (R_avg / Rij) * (Pij - Bij) + B_avg
    
    return Cij


cal_path = 'D:/Calibration/Data'
# generator_0_file = glob.glob(f'{cal_path}/Malus*1001_10_57*.h5')
# generator_90_file = glob.glob(f'{cal_path}/Malus*1007_08_30*.h5')
# generator_45_file = glob.glob(f'{cal_path}/Malus*1007_11_33*.h5')
# generator_135_file = glob.glob(f'{cal_path}/Malus*1007_10_40*.h5')

generator_0_file = glob.glob(f'{cal_path}/Malus*1118_15_15_58*.h5')
generator_90_file = glob.glob(f'{cal_path}/Malus*1118_15_57_15*.h5')
generator_45_file = glob.glob(f'{cal_path}/Malus*1118_15_51_41*.h5')
generator_135_file = glob.glob(f'{cal_path}/Malus*1118_16_02_07*.h5')

g0 = h5py.File(generator_0_file[0],'r+')
g90 = h5py.File(generator_90_file[0],'r+')
g45 = h5py.File(generator_45_file[0],'r+')
g135 = h5py.File(generator_135_file[0],'r+')

#Horizontal Measurements generator, analyzer
runs0 = g0["Measurement_Metadata"].attrs['Runs for each angle']
P_00 = np.mean(g0["P_0 Measurements/UV Raw Images"][:].reshape(runs0,2848,2848),axis=0)
P_090 = np.mean(g0["P_90 Measurements/UV Raw Images"][:].reshape(runs0,2848,2848),axis=0)
P_045 = np.mean(g0["P_45 Measurements/UV Raw Images"][:].reshape(runs0,2848,2848),axis=0)
P_0135 = np.mean(g0["P_135 Measurements/UV Raw Images"][:].reshape(runs0,2848,2848),axis=0)

#Corrected Horizontal
C_00 = correct_img(P_00,Rij[0],Bij[0])
C_090 = correct_img(P_090,Rij[90],Bij[90])
C_045 = correct_img(P_045,Rij[45],Bij[45])
C_0135 = correct_img(P_0135,Rij[135],Bij[135])

fig=plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.title("P0")
plt.imshow(C_00, cmap='gray',interpolation = 'None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,2)
plt.title("P90")
plt.imshow(C_090, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,3)
plt.title("P45")
plt.imshow(C_045, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)


plt.subplot(1,4,4)
plt.title("P135")
plt.imshow(C_0135, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

fig.suptitle("Cij Images — Horizontal Generator", fontsize=18, y=0.8)
plt.tight_layout()
plt.show()


#Vertical Measurements generator, analyzer
runs90 = g90["Measurement_Metadata"].attrs['Runs for each angle']
P_900 = np.mean(g90["P_0 Measurements/UV Raw Images"][:].reshape(runs90,2848,2848),axis=0)
P_9090 =np.mean( g90["P_90 Measurements/UV Raw Images"][:].reshape(runs90,2848,2848),axis=0)
P_9045 = np.mean(g90["P_45 Measurements/UV Raw Images"][:].reshape(runs90,2848,2848),axis=0)
P_90135 = np.mean(g90["P_135 Measurements/UV Raw Images"][:].reshape(runs90,2848,2848),axis=0)


#Corrected Vertical
C_900 = correct_img(P_900,Rij[0],Bij[0])
C_9090 = correct_img(P_9090,Rij[90],Bij[90])
C_9045 = correct_img(P_9045,Rij[45],Bij[45])
C_90135 = correct_img(P_90135,Rij[135],Bij[135])

fig=plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.title("P0")
plt.imshow(C_900, cmap='gray',interpolation = 'None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,2)
plt.title("P90")
plt.imshow(C_9090, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,3)
plt.title("P45")
plt.imshow(C_9045, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)


plt.subplot(1,4,4)
plt.title("P135")
plt.imshow(C_90135, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

fig.suptitle("Cij Images — Vertical Generator", fontsize=18, y=0.8)
plt.tight_layout()
plt.show()

#45 Measurements generator, analyzer
runs45 = g45["Measurement_Metadata"].attrs['Runs for each angle']
P_450 = np.mean(g45["P_0 Measurements/UV Raw Images"][:].reshape(runs45,2848,2848),axis=0)
P_4590 =np.mean( g45["P_90 Measurements/UV Raw Images"][:].reshape(runs45,2848,2848),axis=0)
P_4545 = np.mean(g45["P_45 Measurements/UV Raw Images"][:].reshape(runs45,2848,2848),axis=0)
P_45135 = np.mean(g45["P_135 Measurements/UV Raw Images"][:].reshape(runs45,2848,2848),axis=0)

#Corrected 45
C_450 = correct_img(P_450,Rij[0],Bij[0])
C_4590 = correct_img(P_4590,Rij[90],Bij[90])
C_4545 = correct_img(P_4545,Rij[45],Bij[45])
C_45135 = correct_img(P_45135,Rij[135],Bij[135])

fig=plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.title("P0")
plt.imshow(C_450, cmap='gray',interpolation = 'None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,2)
plt.title("P90")
plt.imshow(C_4590, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,3)
plt.title("P45")
plt.imshow(C_4545, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)


plt.subplot(1,4,4)
plt.title("P135")
plt.imshow(C_45135, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

fig.suptitle("Cij Images — 45 deg Generator", fontsize=18, y=0.8)
plt.tight_layout()
plt.show()

#135 Measurements generator, analyzer
runs135 = g135["Measurement_Metadata"].attrs['Runs for each angle']
P_1350 = np.mean(g135["P_0 Measurements/UV Raw Images"][:].reshape(runs135,2848,2848),axis=0)
P_13590 =np.mean( g135["P_90 Measurements/UV Raw Images"][:].reshape(runs135,2848,2848),axis=0)
P_13545 = np.mean(g135["P_45 Measurements/UV Raw Images"][:].reshape(runs135,2848,2848),axis=0)
P_135135 = np.mean(g135["P_135 Measurements/UV Raw Images"][:].reshape(runs135,2848,2848),axis=0)

#Corrected 135
C_1350 = correct_img(P_1350,Rij[0],Bij[0])
C_13590 = correct_img(P_13590,Rij[90],Bij[90])
C_13545 = correct_img(P_13545,Rij[45],Bij[45])
C_135135 = correct_img(P_135135,Rij[135],Bij[135])

fig=plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.title("P0")
plt.imshow(C_1350, cmap='gray',interpolation = 'None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,2)
plt.title("P90")
plt.imshow(C_13590, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

plt.subplot(1,4,3)
plt.title("P45")
plt.imshow(C_13545, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)


plt.subplot(1,4,4)
plt.title("P135")
plt.imshow(C_135135, cmap='gray',interpolation ='None',vmin=0,vmax=3000)
plt.colorbar(shrink=0.5)

fig.suptitle("Cij Images — 135 deg Generator", fontsize=18, y=0.8)
plt.tight_layout()
plt.show()

flux_matrix = np.array([[C_00,C_090,C_045,C_0135],[C_900,C_9090,C_9045,C_90135],
                        [C_450,C_4590,C_4545,C_45135],
                        [C_1350,C_13590,C_13545,C_135135]
                        ]).reshape(4,4,2848*2848)

W = flux_matrix.T@np.linalg.pinv(Stokes_ideal)
W = W.reshape(2848,2848,4,3)

W =  W / W[..., :, :1]


#np.save('D:/ULTRASIP_Wmatrix.npy', W)
