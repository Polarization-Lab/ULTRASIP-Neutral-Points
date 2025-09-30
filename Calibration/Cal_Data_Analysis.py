# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 12:20:38 2025

@author: Clarissa M. DeLeon 
Analyze and Visualize Calibration Data (not applied to measurements)
NUC: Create Slope and intercept images and save to hdf file - then  
analyze aolp and dolp images

Malus: Calculate average flux images then make intensity over polarizer angle 
plot and compare to original - calculate RMSE (?)
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

# Define model: I = cos^2(theta - theta0)
def malus_fixed(theta_deg, theta0_deg):
    theta_rad = np.deg2rad(theta_deg - theta0_deg)
    return np.cos(theta_rad)**2

# --- Config ---
cal_type = 'NUC'  # 'NUC' or 'Malus'
cal_path = 'D:/Calibration/Data'
cal_files = glob.glob(f'{cal_path}/{cal_type}*.h5')

idx = 7#len(cal_files) - 1  # choose file index #8,7,6,5
Ni, Nj = 2848, 2848       # image size

with h5py.File(cal_files[idx], 'r+') as f:

    if cal_type == 'NUC':
        # --- Load exposures once ---
        exp_times = f['P_0 Measurements/Exposure Times'][:]

        # --- Loop over angles ---
        angles = [0, 45, 90, 135]
        Rij, Bij = {}, {}

        for ang in angles:
            uvimgs = f[f'P_{ang} Measurements/UV Raw Images'][:]
            meas = uvimgs.reshape(len(exp_times), Ni, Nj)

            # Linear regression via least squares (vectorized)
            A = np.vstack([exp_times, np.ones_like(exp_times)]).T  # shape (n,2)
            coeffs, _, _, _ = np.linalg.lstsq(A, meas.reshape(len(exp_times), -1), rcond=None)
            slope = coeffs[0].reshape(Ni, Nj)
            intercept = coeffs[1].reshape(Ni, Nj)

            Rij[ang], Bij[ang] = slope, intercept

        print('Saving...')

        # --- Save to HDF5 ---
        if 'NUC Images' not in f: 
            nuc = f.create_group("NUC Images")
        else:
            del f["NUC Images"]
            nuc = f.create_group("NUC Images")
        for ang in angles:
            nuc.create_dataset(f'P{ang} Rij', data=Rij[ang])
            nuc.create_dataset(f'P{ang} Bij', data=Bij[ang])
            
            plt.figure()
            plt.imshow(Rij[ang],cmap='gray',vmin=0,vmax=0.003)
            plt.title(f'P{ang} Rij')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.figure()
            plt.imshow(Bij[ang],cmap='gray',vmin=0,vmax=100)
            plt.title(f'P{ang} Bij')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])

        # --- Example test correction (optional demo) ---
        with h5py.File(cal_files[5], 'r+') as test_file:
            run = 119
            exp_times = test_file['P_0 Measurements/Exposure Times'][:]

            test_imgs = {}
            for ang in angles:
                test_uvimgs = test_file[f'P_{ang} Measurements/UV Raw Images'][:]
                test_imgs[ang] = test_uvimgs.reshape(len(exp_times), Ni, Nj)[run, :, :]

            # Get global averages
            R_avg = {ang: np.mean(Rij[ang]) for ang in angles}
            B_avg = {ang: np.mean(Bij[ang]) for ang in angles}

            # Apply correction
            Cij = {}
            for ang in angles:
                Cij[ang] = (R_avg[ang] / Rij[ang]) * (test_imgs[ang] - Bij[ang]) + B_avg[ang]
    
            Cstack = np.stack([Cij[0], Cij[90], Cij[45], Cij[135]], axis=0) 
            Stokes = np.linalg.pinv(W)@(Cstack.reshape(4, 2848*2848))
            Stokes = Stokes.reshape(3, 2848, 2848)

            I, Q, U = Stokes
    
            dolp = (np.sqrt(Q**2 + U**2) / I)*100
            dolp_mean = np.average(dolp)
            dolp_std = np.std(dolp)
            dolp_median  = np.median(dolp)
    
            aolp = 0.5 * np.arctan2(U, Q)
            aolp = np.mod(np.degrees(aolp),180)
    
            #Averages
            avgQ = np.flip(np.average(Q/I,axis=1)) #row avg
            avgU = np.average(U/I,axis=0)#col avg

            dolp_avg = np.sqrt((avgQ**2)+(avgU**2))*100
            dolpavg_mean = np.average(dolp_avg)
            dolpavg_std = np.std(dolp_avg)
            dolpavg_median = np.median(dolp_avg)

    
            #-------Plots--------------------------------------------------------------
    
            # 2x2 grid of Cij images
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.subplots_adjust(top=0.88, wspace=0.05, hspace=0.15)

            # Flatten axes array for easy iteration
            axes = axes.ravel()

            # Normalize colormap limits across all Cij images
            vmin = min(np.min(Cij[ang]) for ang in angles)
            vmax = max(np.max(Cij[ang]) for ang in angles)

            for idx, ang in enumerate(angles):
                im = axes[idx].imshow(Cij[ang], cmap='gray', vmin=vmin, vmax=vmax)
                axes[idx].set_title(f"P{ang}", fontsize=14)
                axes[idx].set_xticks([]); axes[idx].set_yticks([])

            # Shared colorbar along the top
            cbar_ax = fig.add_axes([0.14, 0.94, 0.75, 0.02])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=12)

            # Suptitle with exposure time
            fig.suptitle(f"Cij Images — Exposure Time = {exp_times[run]:.6f} us", fontsize=18, y=1.03)

            plt.show()

            fig, axes = plt.subplots(1, 3, figsize=(16, 4))  # 1 row, 4 columns
    
            #Plot I
            im0 = axes[0].imshow(I, cmap='gray', vmin=0, vmax=np.max(I), interpolation = 'None')
            axes[0].set_title('I',fontsize=20)
            cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            cbar0.ax.tick_params(labelsize=14)
            axes[0].set_xticks([]); axes[0].set_yticks([])
    
            # Plot Q/I
            im1 = axes[1].imshow(Q/I, cmap=cmo.curl, interpolation = 'None',vmin=-0.1,vmax=0.1)
            axes[1].set_title('Q/I',fontsize=20)
            cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(labelsize=14)
            axes[1].set_xticks([]); axes[1].set_yticks([])

            # Plot U/I
            im2 = axes[2].imshow(U/I, cmap=cmo.curl, interpolation = 'None',vmin=-0.1,vmax=0.1)
            axes[2].set_title('U/I',fontsize=20)
            cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(labelsize=14)
            axes[2].set_xticks([]); axes[2].set_yticks([])

            plt.tight_layout()
            plt.show()

            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

            # Image plot
            im = axes[0].imshow(aolp, cmap=cmo.phase, interpolation='none', vmin=0, vmax=180)
            cbar = fig.colorbar(im, ax=axes[0])
            cbar.ax.tick_params(labelsize=14)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            # Histogram plot
            axes[1].hist(aolp.flatten(), bins=50, edgecolor='black',range=(0,180))

            axes[1].set_ylabel('Frequency', fontsize=14)
            axes[1].tick_params(axis='both', labelsize=12)

            plt.suptitle('AoLP Corrected [deg]', fontsize=20)
            plt.tight_layout()
            plt.show()
        
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
            
            # Left: DoLP image
            im = axes[0].imshow(dolp, cmap='hot', interpolation='none', vmin=0, vmax=10)
            cbar = fig.colorbar(im, ax=axes[0])
            cbar.ax.tick_params(labelsize=14)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            # Right: histogram
            axes[1].hist(dolp.flatten(), bins=50, edgecolor='black',range=(0,15))
            axes[1].set_ylabel('Frequency', fontsize=14)
            axes[1].tick_params(axis='both', labelsize=12)
            
            # Add text box with mean and std in upper right
            textstr = f"Mean = {dolp_mean:.2f}%\nStd = {dolp_std:.2f}%\nMed = {dolp_median:.2f}%"
            axes[1].text(0.95, 0.95, textstr, transform=axes[1].transAxes,
                         fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

            plt.suptitle('DoLP Corrected [%]', fontsize=20)
            plt.tight_layout()
            plt.show()
    
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Scatter plot
            ax.scatter(dolp_avg, range(len(dolp_avg)), color='green')
            ax.set_xlabel(r'$DoLP_{rc} [\%]$', fontsize=15)
            ax.set_ylabel('Pixel Index', fontsize=15)
            ax.set_title(r'DoLP from $\bar{c}_{U},\bar{r}_{Q}$', fontsize=16)
            
            # Add text box to the right of plot
            textstr = f"Mean = {dolpavg_mean:.2f}%\nStd = {dolpavg_std:.2f}%\nMed = {dolpavg_median:.2f}%"
            ax.text(1.05, 0.5, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

            plt.tight_layout()
            plt.show()
    
    if cal_type == 'Malus':
        # Load dataset for each angle 
        angles = np.r_[0:363:3]  # 0, 3, 6, ..., 360
        avg_intensity = []
        gen_ang = f['Measurement_Metadata'].attrs['Angle of Generator Linear Polarizer']
        for angle in angles:
            # Construct dataset name dynamically
            dataset_name = f"P_{angle} Measurements/UV Raw Images"
    
            # Load and reshape (5, 2848, 2848)
            data = f[dataset_name][:].reshape(5, 2848, 2848)
    
            # Average across the 5 runs
            mean_image = np.mean(data, axis=0)
            
            if angle in [0,45,90,135]:
                plt.figure()
                plt.imshow(mean_image,cmap='gray',vmin=0,vmax=2300)
                plt.title(f'P_gen: {gen_ang}, Avg P_{angle}')
                plt.colorbar()
    
            # Compute overall average pixel value
            avg_val = np.mean(mean_image)
    
            # Store result
            avg_intensity.append(avg_val)
            
        norm_intensity = avg_intensity/np.max(avg_intensity)

        # Plot average intensity vs angle
        plt.figure(figsize=(8, 5))
        plt.plot(angles,norm_intensity, 'o-', color='darkblue')
        plt.xlabel('Analyzer Linear Polarizer Angle (degrees)', fontsize=12)
        plt.ylabel('Average Pixel Value', fontsize=12)
        plt.title(f'Generator Angle: {gen_ang}', fontsize=14)
        plt.grid(True)
        
        
        plt.xticks(np.arange(0, 361, 45))
        plt.yticks(np.arange(0,1.25,0.25))
        plt.show()
        
        # Initial guess: angle of maximum intensity
        theta0_guess = angles[np.argmax(norm_intensity)]

        # Fit only theta0
        popt, pcov = curve_fit(malus_fixed, angles, norm_intensity, p0=[theta0_guess])
        theta0_fit = popt[0]
        theta0_err = np.sqrt(np.diag(pcov))[0]

        # Theoretical fit
        fit_intensity = malus_fixed(angles, theta0_fit)

        fig, ax = plt.subplots(figsize=(8,5))

        # Plot measured normalized intensity
        ax.plot(angles, norm_intensity, 'o-', color='darkblue', label='Measured')

        # Plot fitted Malus curve
        ax.plot(angles, fit_intensity, '--', color='red',
        label=fr'Fit: $\theta_0$ = {theta0_fit-180:.2f}°')

        # Labels, title, legend
        ax.set_xlabel('Analyzer Linear Polarizer Angle (degrees)', fontsize=14)
        ax.set_ylabel('Normalized Intensity', fontsize=14)
        ax.set_title(f'Generator Angle: {gen_ang}', fontsize=14)
        ax.legend(fontsize=12,loc='upper left')
        ax.grid(True)
        plt.xticks(np.arange(0, 361, 45))
        ax.set_yticks(np.arange(0, 1.25, 0.25))
    
        plt.tight_layout()
        plt.show()
