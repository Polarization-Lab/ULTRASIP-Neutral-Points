# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:41:58 2026

@author: ULTRASIP_1
"""

#Import libraries 
import matplotlib.pyplot as plt
import numpy as np
import cmocean.cm as cmo
import glob
import h5py
import os

malus_data = {}
#Correct image
def correct_img(Pij,Rij,Bij):
    R_avg = np.mean(Rij)
    B_avg = np.mean(Bij)
    Cij = (R_avg / Rij) * (Pij - Bij) + B_avg
    
    return Cij

def create_analyzervec(gen_angle):
    
    Stokes_vec = np.array([1, np.cos(2*gen_angle), np.sin(2*gen_angle)])
    
    return Stokes_vec 

#NUC
data = np.load('D:/NUC_0813.npz', allow_pickle=True)

Rij = data['arr1'].item()   # convert array-object → Python dict
Bij = data['arr2'].item()

#Polarized MEasurements
cal_path = 'D:/Calibration/Data'
files = glob.glob(f'{cal_path}/Malus*20251124_*.h5')


for step in [None, 90, 75, 60, 45, 30, 15]:
    
    malus_data = {}

    if step is None:
        gen_angles = np.array([0, 45, 90, 135])
    else:
        gen_angles = np.arange(0, 361, step)
        
    Stokes_ideal = np.column_stack([create_analyzervec(a) for a in np.radians(gen_angles)])

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
            
            gen = float(meta.attrs["Angle of Generator Linear Polarizer"])
            runs      = int(meta.attrs["Runs for each angle"])

            
            if gen in gen_angles:
                # --- Print info ---
                print(f"{os.path.basename(f)} → angle = {gen}, runs = {runs}")
        
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
                malus_data[gen] = {
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
                continue
        
    W = Cmat.T@np.linalg.pinv(Stokes_ideal)
    W = W.reshape(2848,2848,4,3)
    # compute singular values for each 4x3 matrix
    u, s, vh = np.linalg.svd(W, full_matrices=False)

    # condition number = largest singular value / smallest singular value
    cond_W = s[..., 0] / s[..., -1]

    print(cond_W.shape)   # (2848, 2848)
    
    outliers = np.where(cond_W > 1.5)
    num_pixels_above = np.size(outliers,1)/(2848*2848)
    print(num_pixels_above*100)
    
    lowers = np.where(cond_W < 1.4)
    num_pixels_below = np.size(lowers,1)/(2848*2848)
    print(num_pixels_below*100)
    
    fig, ax = plt.subplots(figsize=(7,6))

    im = ax.imshow(cond_W, interpolation='None',cmap='coolwarm',vmin=1.4,vmax=1.5)

    cbar = fig.colorbar(im, ax=ax,shrink=0.8)
    cbar.set_label('Condition Number', fontsize=14)   # colorbar TITLE size
    cbar.ax.tick_params(labelsize=12)                             # colorbar TICK size

    ax.set_xlabel('Pixel', fontsize=14)
    ax.set_ylabel('Pixel', fontsize=14)

    plt.tight_layout()
    plt.show()
    
    np.save(f'D:/ULTRASIP_Wmatrix_{step}.npy', W)

    