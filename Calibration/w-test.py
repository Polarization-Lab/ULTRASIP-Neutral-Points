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

gen_angles = np.r_[0:365:15]
Stokes_ideal = np.column_stack([create_analyzervec(a) for a in np.radians(gen_angles)])


cal_path = 'D:/Calibration/Data'
files = glob.glob(f'{cal_path}/Malus*20251124_*.h5')

# Dictionary to store all results
# keys = generator angle (float)
# values = dict with P0, P90, P45, P135 averaged images
malus_data = {}

for f in files:
    with h5py.File(f, 'r') as h:

        # --- Read metadata ---
        if "Measurement_Metadata" not in h:
            print(f"{os.path.basename(f)} → 'Measurement_Metadata' missing")
            continue

        meta = h["Measurement_Metadata"]

        if "Angle of Generator Linear Polarizer" not in meta.attrs:
            print(f"{os.path.basename(f)} → angle attribute missing")
            continue

        gen_angle = float(meta.attrs["Angle of Generator Linear Polarizer"])
        runs      = int(meta.attrs["Runs for each angle"])

        # --- Print info ---
        print(f"{os.path.basename(f)} → angle = {gen_angle}, runs = {runs}")

        # --- Read measurement groups ---
        P0_stack   = h["P_0 Measurements/UV Raw Images"][:]
        P90_stack  = h["P_90 Measurements/UV Raw Images"][:]
        P45_stack  = h["P_45 Measurements/UV Raw Images"][:]
        P135_stack = h["P_135 Measurements/UV Raw Images"][:]

        # Confirm size = runs × 2848 × 2848
        # Average over the run dimension
        P0   = np.mean(P0_stack.reshape(runs, 2848, 2848), axis=0)
        P90  = np.mean(P90_stack.reshape(runs, 2848, 2848), axis=0)
        P45  = np.mean(P45_stack.reshape(runs, 2848, 2848), axis=0)
        P135 = np.mean(P135_stack.reshape(runs, 2848, 2848), axis=0)

                # --- apply correction ---
        C0   = correct_img(P0,   Rij[0], Bij[0])
        C90  = correct_img(P90,  Rij[90], Bij[90])
        C45  = correct_img(P45,  Rij[45], Bij[45])
        C135 = correct_img(P135, Rij[135], Bij[135])

        # --- store everything ---
        malus_data[gen_angle] = {
            "C0": C0, "C90": C90, "C45": C45, "C135": C135,
            "runs": runs,
            "filename": os.path.basename(f),
        }


        # Sort angles so row order is consistent
        angles = sorted(malus_data.keys())

        # Build matrix with rows = angles, cols = analyzers
        Cmat = np.array([
            [
                malus_data[a]["C0"],
                malus_data[a]["C90"],
                malus_data[a]["C45"],
                malus_data[a]["C135"]
            ]
            for a in angles
                ])
        
        
W = Cmat.T@np.linalg.pinv(Stokes_ideal)
W = W.reshape(2848,2848,4,3)

np.save('D:/ULTRASIP_Wmatrix_mas.npy', W)

