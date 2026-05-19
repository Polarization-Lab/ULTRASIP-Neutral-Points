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
date = '2025_10_23'

#Datapath
basepath = 'D:/Data'
#basepath = 'C:/Users/ULTRASIP_1/OneDrive/Desktop'
folderdate = os.path.join(basepath,date)
files = glob.glob(f'{folderdate}/*.h5')


for file in files:
    try:
        with h5py.File(file, 'r+') as f:
            
            # Skip if dataset/group already exists
            if "Corrected Neutral Point Localization" in f:
                del f['Corrected Neutral Point Localization']
                continue
            if 'Neutral Point Estimation' in f:
                print(f"{file} has Neutral Point Estimation")
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

                saz = aq['UV Image Data/sun_az_corrected'][()]
                sza = aq['UV Image Data/sun_zen_corrected'][()]



                vza = aq["UV Image Data/view_zen_corrected"][:]
                vaz = aq["UV Image Data/view_az_corrected"][:]

                vza = vza[:,0]
                vaz = vaz[0,:]

                q = Q/I
                u=U/I

                dolp = np.sqrt((q**2)+(u**2))*100
                
                aolp = 0.5*np.arctan2(U,Q)
                aolp = np.mod(np.degrees(aolp),180)
            
            
            # Convert to slice objects
                q_start, q_stop = map(int, q_cropped_region.split(':'))
                u_start, u_stop = map(int, u_cropped_region.split(':'))

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
               
            # Get the fitted values and residuals
                qfit_line = qresults.fittedvalues
                qresiduals = vza_crop - qfit_line

                qslope = qresults.params[1]
                qint = qresults.params[0]
                qint_stderror = qresults.bse[0]

            # Calculate weights (inverse of standard deviation)
                weights = (1 / np.std(vaz_crop)) * np.ones_like(vaz_crop)
               
            # Add a constant (intercept) to the independent variable
                avg_u_with_intercept = sm.add_constant(avgu)
               
            # Weighted least squares regression
                model = sm.WLS(vaz_crop, avg_u_with_intercept, weights=weights)
                uresults = model.fit()
            # print(qresults.summary())
            # print(uresults.summary())

            # Get the fitted values and residuals
                ufit_line = uresults.fittedvalues
                residuals = vaz_crop - ufit_line

                uslope = uresults.params[1]
                uint = uresults.params[0]
                uint_stderror = uresults.bse[0]
            
                print(uint,qint)
            
                np_est_corr = f.create_group("Corrected Neutral Point Localization")
                np_est_corr.create_dataset('Corrected Estimation NP Location (zen,az) [deg]', data = np.array([qint, uint]))
                np_est_corr.create_dataset('Sun Location (zen,az) [deg]', data = np.array([sza, saz]))
                np_est_corr.attrs['Corrected Zenith Error [arcseconds]'] = qint_stderror * 3600
                np_est_corr.attrs['Corrected Azimuth Error [arcseconds]'] = uint_stderror * 3600
                np_est_corr.attrs['Q Cropped Region'] = q_cropped_region
                np_est_corr.attrs['U Cropped Region'] = u_cropped_region
                np_est_corr.attrs['Aquisition Number'] = aquisition_number
                print("Neutral point estimation correction saved.")

    except (OSError, IOError) as e:
        print(f"Could not open file: {file} | Error: {e}")
        continue
