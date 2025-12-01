# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:13:01 2025

@author: ULTRASIP_1
"""
#Import libraries 
import matplotlib.pyplot as plt
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

def create_analyzervec(gen_angle):
    
    Stokes_vec = np.array([1, np.cos(2*gen_angle), np.sin(2*gen_angle)])
    
    return Stokes_vec 

gen_angles = np.r_[0:365:45]
Stokes_ideal = np.column_stack([create_analyzervec(a) for a in np.radians(gen_angles)])


cal_path = 'D:/Calibration/Data'
