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
from matplotlib.patches import Rectangle
import pandas as pd
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

results = []

#------------------------Load Observation---------------------------------------
#Load observations 
#Set Date of Measurements 
date = '2025_10_23'

#Datapath
basepath = 'D:/Data'
#basepath = 'C:/Users/ULTRASIP_1/OneDrive/Desktop'
folderdate = os.path.join(basepath,date)
files = glob.glob(f'{folderdate}/*.h5')
idx = np.arange(0,len(files))

for i in idx:
    f = h5py.File(files[i],'r+')
    
    if "Neutral Point Estimation" in f:
        np_est = f['Neutral Point Estimation']
        # Convert attributes to variables
        for name, value in np_est.attrs.items():
            # make valid variable name
            var_name = name.replace(" ", "_").replace("-", "_").lower()
            # assign variable
            globals()[var_name] = value
    
        for aqnum in range(0,len(f.keys())-1):
            
            if f"Aquistion_{aqnum}" in f:

                aq = f[f'Aquistion_{aqnum}']

                I = aq["UV Image Data/I_corrected"][:]
                Q = aq["UV Image Data/Q_corrected"][:]
                U = aq["UV Image Data/U_corrected"][:]
            
                plt.figure(figsize=(16, 8))
                plt.imshow(Q/I,cmap=colmap,interpolation='None',vmin=-0.03,vmax=0.03)
                plt.title('Q/I')
                plt.colorbar()

                plt.figure(figsize=(16, 8))
                plt.imshow(U/I,cmap=colmap,interpolation='None',vmin=-0.03,vmax=0.03)
                plt.title('U/I')
                plt.colorbar()

                q = Q/I
                u=U/I

                vza = aq["UV Image Data/view_zen"][:]
                vaz = aq["UV Image Data/view_az"][:]

                vza = vza[:,0]
                vaz = vaz[0,:]

                sun_az = aq['UV Image Data/sun_az'][()]


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

                q = Q/I
                u=U/I

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
                
                plt.text(-0.02, (np.median(vza_crop)+2),
                         f'Intercept: {qint:.4f}$^\circ$ \n $\sigma_{{zen}}$: {qint_stderror:.4f} arcsec',
                         fontsize=20,
                         bbox=dict(facecolor='lightgray', alpha=1))

                # Add labels and legend
                plt.ylabel(r'Zenith [$\circ$]',fontsize=20)
                plt.xlabel(r'$\bar{r_{Q}}$', fontsize = 20)
                plt.xlim(-0.2, 0.2)
                #plt.ylim([50.5, 54.5])
                plt.yticks(fontsize=18)
                plt.xticks(fontsize=18)
                plt.gca().invert_yaxis() 
                #plt.title('Weighted Linear Regression with Fit Error')
                plt.grid(True)
                #plt.legend(fontsize=20,loc='upper left')
                plt.show()


                # ---- Figure 2: U vs Zenith ----
                plt.figure(figsize=(12, 8))
                plt.scatter(avgu, vaz_crop, color='green')
                plt.plot(avgu, ufit_line, color='gold', label='Weighted fitted line', linewidth=5)
                plt.axvline(x=0, lw=5, color='red', zorder=0)
                plt.axhline(y=sun_az, lw=3, color='purple')


                plt.text(0.007, (np.median(vaz_crop)+2),
                         f'Intercept: {uint:.4f}$^\circ$ \n $\sigma_{{az}}$: {uint_stderror:.4f} arcsec',
                         fontsize=20,
                         bbox=dict(facecolor='lightgray', alpha=1))

                # Add labels and legend
                plt.ylabel(r'Azimuth [$\circ$]',fontsize=20)
                plt.xlabel(r'$\bar{c_{U}}$', fontsize = 20)
                plt.xlim(-0.2, 0.2)
                #plt.ylim([50.5, 54.5])
                plt.yticks(fontsize=18)
                plt.xticks(fontsize=18)
                #plt.title('Weighted Linear Regression with Fit Error')
                plt.grid(True)
                #plt.legend(fontsize=20,loc='upper left')
                plt.show()

                print("Abs Azimuth Difference",np.abs(sun_az-uint))
                
                results.append({
                    "file": os.path.basename(files[i]),
                    "acquisition": aqnum,                 
                    "sun_az_deg": float(sun_az),
                    "neutral_az_deg": float(uint),
                    "abs_az_diff_deg": float(np.abs(sun_az - uint)),
                    "u_r2": float(uresults.rsquared),

                    })
            else:
                continue

    else:   
            continue
        
df = pd.DataFrame(results)
outfile = os.path.join(folderdate, f"{date}_neutral_point_fit_rotation_analysis.csv")
df.to_csv(outfile, index=False)