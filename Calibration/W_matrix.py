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

cal_path = 'D:/Calibration/Data'
generator_0_file = glob.glob(f'{cal_path}/Malus*1001_10_57*.h5')
generator_90_file = glob.glob(f'{cal_path}/Malus*1007_08_30*.h5')
generator_45_file = glob.glob(f'{cal_path}/Malus*1007_11_33*.h5')
generator_135_file = glob.glob(f'{cal_path}/Malus*1007_10_40*.h5')

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


#Vertical Measurements generator, analyzer
runs90 = g90["Measurement_Metadata"].attrs['Runs for each angle']
P_900 = np.mean(g90["P_0 Measurements/UV Raw Images"][:].reshape(runs90,2848,2848),axis=0)
P_9090 =np.mean( g90["P_90 Measurements/UV Raw Images"][:].reshape(runs90,2848,2848),axis=0)
P_9045 = np.mean(g90["P_45 Measurements/UV Raw Images"][:].reshape(runs90,2848,2848),axis=0)
P_90135 = np.mean(g90["P_135 Measurements/UV Raw Images"][:].reshape(runs90,2848,2848),axis=0)

#45 Measurements generator, analyzer
runs45 = g45["Measurement_Metadata"].attrs['Runs for each angle']
P_450 = np.mean(g45["P_0 Measurements/UV Raw Images"][:].reshape(runs45,2848,2848),axis=0)
P_4590 =np.mean( g45["P_90 Measurements/UV Raw Images"][:].reshape(runs45,2848,2848),axis=0)
P_4545 = np.mean(g45["P_45 Measurements/UV Raw Images"][:].reshape(runs45,2848,2848),axis=0)
P_45135 = np.mean(g45["P_135 Measurements/UV Raw Images"][:].reshape(runs45,2848,2848),axis=0)

#135 Measurements generator, analyzer
runs135 = g135["Measurement_Metadata"].attrs['Runs for each angle']
P_1350 = np.mean(g135["P_0 Measurements/UV Raw Images"][:].reshape(runs135,2848,2848),axis=0)
P_13590 =np.mean( g135["P_90 Measurements/UV Raw Images"][:].reshape(runs135,2848,2848),axis=0)
P_13545 = np.mean(g135["P_45 Measurements/UV Raw Images"][:].reshape(runs135,2848,2848),axis=0)
P_135135 = np.mean(g135["P_135 Measurements/UV Raw Images"][:].reshape(runs135,2848,2848),axis=0)

flux_matrix = np.array([[P_00,P_090,P_045,P_0135],[P_900,P_9090,P_9045,P_90135],
                        [P_450,P_4590,P_4545,P_45135],
                        [P_1350,P_13590,P_13545,P_135135]
                        ]).reshape(4,4,2848*2848)

W = flux_matrix.T@np.linalg.pinv(Stokes_ideal)
W = W.reshape(2848,2848,4,3)

np.save('ULTRASIP_Wmatrix.npy', W)
