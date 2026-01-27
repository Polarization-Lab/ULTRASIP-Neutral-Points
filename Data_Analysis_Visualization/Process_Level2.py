# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 13:47:25 2025

@author: C.M.DeLeon
"""

#Import Libraries 
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cmocean.cm as cmo
import glob,os
from scipy.stats import norm
import matplotlib.mlab as mlab
import statsmodels.api as sm
from scipy.optimize import curve_fit

#Load observations 
#Set Date of Measurements 
date = '2025_06_10'

#Datapath
basepath = 'D:/Data'
#basepath = 'C:/Users/ULTRASIP_1/OneDrive/Desktop'
folderdate = os.path.join(basepath,date)
files = glob.glob(f'{folderdate}/*.h5')
i = len(files) # Set file index you want to view - default is set to the last one (len(files)-1)

for idx in range(0, 1):
    print(f'Processing file {idx} of {i}: {files[idx]}')
    
    try:
        f = h5py.File(files[idx], 'r+')

        for aqnum in range(0,len(f.keys())-1):
    
            timestamp = files[idx][32:46]
            #Set aquisition to view
            aq = f[f'Aquistion_{aqnum}']
            print(aqnum)
    
            if 'Neutral Point Estimation' in aq:
                del aq['Neutral Point Estimation']
                
            #Load geometry 
            view_az = aq['UV Image Data/view_az'][:]
            view_zen = aq['UV Image Data/view_zen'][:]
    
            #Load polarized data products
            I = aq['UV Image Data/I_corrected'][:]
            Q = aq['UV Image Data/Q_corrected'][:]
            U = aq['UV Image Data/U_corrected'][:]
    
            q = Q/I
            u = U/I
    
            dolp = np.sqrt(q**2+u**2)*100
            aolp = 0.5*np.arctan2(U,Q)
            aolp = np.mod(np.degrees(aolp),180)


            #axis 0 is along columns, axis=1 is along rows
            avgq = np.flip(np.average(q,axis=1)) #row avg
            avgu = np.average(u,axis=0)#col avg
            

            #HI :)
            
            vza = view_zen[:,0]
            vaz = view_az[0,:]
                        
            #rows,colummns for subfigs
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))  
            
            # Plot I
            im0 = axes[0,0].imshow(I, cmap='gray', interpolation = 'None',extent=[view_az.min(), view_az.max(), view_zen.max(), view_zen.min()], vmin = 0, vmax = 1)
            axes[0,0].set_title('I',fontsize=20)
            axes[0,0].set_ylabel('Zenith [$\circ$]',fontsize=15)
            axes[0,0].set_xlabel('Azimuth [$\circ$]',fontsize=15)
            plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)
            
            #Plot Q/I
            im1 = axes[0,1].imshow(Q/I, cmap=cmo.curl, vmin=-1, vmax=1 ,extent=[view_az.min(), view_az.max(), view_zen.max(), view_zen.min()], interpolation = 'None')
            axes[0,1].set_title('Q/I',fontsize=20)
            #axes[0,1].set_ylabel('Zenith [$\circ$]',fontsize=15)
            axes[0,1].set_xlabel('Azimuth [$\circ$]',fontsize=15)
            plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)
            
            # Plot U/I
            im2 = axes[0,2].imshow(U/I, cmap=cmo.curl, vmin=-1, vmax=1 ,extent=[view_az.min(), view_az.max(), view_zen.max(), view_zen.min()], interpolation = 'None')
            axes[0,2].set_title('U/I',fontsize=20)
            #axes[0,1].set_ylabel('Zenith [$\circ$]',fontsize=15)
            axes[0,2].set_xlabel('Azimuth [$\circ$]',fontsize=15)
            plt.colorbar(im2, ax=axes[0,2], fraction=0.046, pad=0.04)
            
            # Plot I
            im3 = axes[1,0].hist(I.flatten())
            axes[1,0].set_ylabel('Frequency',fontsize=15)
            axes[1,0].set_xlabel('Value',fontsize=15)
            
            # Plot Q/I
            im3 = axes[1,1].hist((Q/I).flatten())
            #axes[1,1].set_ylabel('Frequency',fontsize=15)
            axes[1,1].set_xlabel('Value',fontsize=15)
            
            # Plot U/I
            im3 = axes[1,2].hist((U/I).flatten())
            #axes[1,2].set_ylabel('Frequency',fontsize=15)
            axes[1,2].set_xlabel('Value',fontsize=15)
            plt.suptitle(f'{timestamp, aqnum}',fontsize=20)
            
            plt.tight_layout()
            plt.show()
            

            #rows,colummns for subfigs
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))  
            
            # Plot DoLP
            im0 = axes[0,0].imshow(dolp, cmap='hot', interpolation = 'None',extent=[view_az.min(), view_az.max(), view_zen.max(), view_zen.min()], vmin = 0, vmax = 1)
            axes[0,0].set_title('DoLP [%]',fontsize=20)
            axes[0,0].set_ylabel('Zenith [$\circ$]',fontsize=15)
            axes[0,0].set_xlabel('Azimuth [$\circ$]',fontsize=15)
            plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)
            
            #Plot log(DoLP)
            im1 = axes[0,1].imshow(np.log(dolp), cmap='Blues_r', vmin=-3, vmax=1.5 ,extent=[view_az.min(), view_az.max(), view_zen.max(), view_zen.min()], interpolation = 'None')
            axes[0,1].set_title('log(DoLP [%])',fontsize=20)
            #axes[0,1].set_ylabel('Zenith [$\circ$]',fontsize=15)
            axes[0,1].set_xlabel('Azimuth [$\circ$]',fontsize=15)
            plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)
            
            # Plot AoLP
            im2 = axes[0,2].imshow(aolp, cmap=cmo.phase,extent=[view_az.min(), view_az.max(), view_zen.max(), view_zen.min()], interpolation = 'None')
            axes[0,2].set_title('AoLP [$\circ$]',fontsize=20)
            #axes[0,2].set_ylabel('Zenith [$\circ$]',fontsize=15)
            axes[0,2].set_xlabel('Azimuth [$\circ$]',fontsize=15)
            plt.colorbar(im2, ax=axes[0,2], fraction=0.046, pad=0.04)
            
            # Plot AoLP
            im3 = axes[1,0].hist(dolp.flatten())
            axes[1,0].set_ylabel('Frequency',fontsize=15)
            axes[1,0].set_xlabel('Value',fontsize=15)
            
            # Plot AoLP
            im3 = axes[1,1].hist(np.log(dolp).flatten())
            #axes[1,1].set_ylabel('Frequency',fontsize=15)
            axes[1,1].set_xlabel('Value',fontsize=15)
            
            # Plot AoLP
            im3 = axes[1,2].hist(aolp.flatten())
            #axes[1,2].set_ylabel('Frequency',fontsize=15)
            axes[1,2].set_xlabel('Value',fontsize=15)
            plt.suptitle(f'{timestamp, aqnum}',fontsize=20)
            
            plt.tight_layout()
            plt.show()
            

            #Averages
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].scatter(avgq,range(len(avgq)),color='green')
            axes[0].axvline(x=0,lw=5,color='red')
            axes[0].set_xlabel(r'$\bar{r}_{Q}$',fontsize=15)
            axes[0].set_ylabel('Pixel value',fontsize=15)
            axes[0].grid()
            #axes[0].set_xlim(-0.02, 0.02)
            #axes[0].set_ylim(55,56)
            
            axes[1].scatter(avgu,range(len(avgu)),color='green')
            axes[1].axvline(x=0,lw=5,color='red')
            axes[1].set_xlabel(r'$\bar{c}_{U}$',fontsize=15)
            axes[1].set_ylabel('Pixel Value',fontsize=15)

            
            plt.tight_layout()
            plt.grid()
            plt.show()
            
            run = input("Cropping? (Yes/No): ")
            if run =='yes' or run == 'Yes' or run == 'YES' or run =='y':
                q_range_str = input("Q range?: ")  # e.g., "1000:2000"
                u_range_str = input("U range?: ")

                # Convert to slice objects
                q_start, q_stop = map(int, q_range_str.split(':'))
                u_start, u_stop = map(int, u_range_str.split(':'))

                avgq = avgq[q_start:q_stop]
                avgu = avgu[u_start:u_stop]

                vza = vza[q_start:q_stop]
                vaz = vaz[u_start:u_stop]
            else:
                continue
            

            #Averages
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].scatter(avgq,vza,color='green')
            axes[0].axvline(x=0,lw=5,color='red')
            axes[0].set_xlabel(r'$\bar{r}_{Q}$',fontsize=15)
            axes[0].set_ylabel('Zenith [deg]',fontsize=15)

            #axes[0].set_xlim(-0.02, 0.02)
            #axes[0].set_ylim(55,56)
            
            axes[1].scatter(avgu,vaz,color='green')
            axes[1].axvline(x=0,lw=5,color='red')
            axes[1].set_xlabel(r'$\bar{c}_{U}$',fontsize=15)
            axes[1].set_ylabel('Azimuth [deg]',fontsize=15)

            
            plt.tight_layout()
            plt.show()
            
  
            run = input("Run neutral point estimation? (Yes/No): ")
            print(f"{run}!")
            
            if run == 'no' or run =='No' or run == 'NO':
                continue 
            if run =='yes' or run == 'Yes' or run == 'YES':
                print('Running Estimation')
                
                saz =  aq['UV Image Data/sun_az'][()]
                sza = aq['UV Image Data/sun_zen'][()]
        
                #---Altitude Estimate------------#

                # Calculate weights (inverse of standard deviation)
                weights = (1 / np.std(vza)) * np.ones_like(vza)
                
                # Add a constant (intercept) to the independent variable
                avg_q_with_intercept = sm.add_constant(avgq)
                
                # Weighted least squares regression
                model = sm.WLS(vza, avg_q_with_intercept, weights=weights)
                results = model.fit()
                
                # Get the fitted values and residuals
                fit_line = results.fittedvalues
                residuals = vza - fit_line
                
                # Plot the fitted line
                plt.figure(figsize=(12,8))
                plt.scatter(avgq,vza,color='green')
                plt.plot(avgq, fit_line, color='gold', label='Weighted fitted line',linewidth=7)
                plt.gca().invert_xaxis()
                # Calculate the standard deviation of the residuals
                std_residuals = np.std(residuals)
                
                # Create error bands
                #plt.fill_between(avgq, fit_line - std_residuals, fit_line + std_residuals, color='red', alpha=0.3, label='Uncertainty region')
                plt.axvline(0,color='red',linewidth=5)
                # Add labels and legend
                plt.ylabel(r'Zenith [$\circ$]',fontsize=20)
                plt.xlabel(r'$\bar{r_{Q}}$', fontsize = 20)
                plt.xlim(-0.025, 0.025)
                #plt.ylim([50.5, 54.5])
                plt.yticks(fontsize=15)
                plt.xticks(fontsize=15)
                plt.gca().invert_yaxis() 
                #plt.title('Weighted Linear Regression with Fit Error')
                plt.grid(True)
                #plt.legend(fontsize=20,loc='upper left')
                plt.show()
                
                # Print regression results
                print(results.summary())
                
                # Print regression results and specific parameters
                print("Slope:", results.params[1])
                print("Intercept:", results.params[0])
                print("R-squared:", results.rsquared)
                print("Standard Error of [Intercept, Slope]:", results.bse)
                alt_slope_error = results.bse[1]
                altitude_error = results.bse[0]
                alt_slope = results.params[1]
                altitude = results.params[0]
                
                
                #---Azimuth Estimate------------#
                
                # Calculate weights (inverse of standard deviation)
                weights = (1 / np.std(vaz)) * np.ones_like(vaz)
                
                # Add a constant (intercept) to the independent variable
                avg_u_with_intercept = sm.add_constant(avgu)
                
                # Weighted least squares regression
                model = sm.WLS(vaz, avg_u_with_intercept, weights=weights)
                results = model.fit()
                
                # Get the fitted values and residuals
                fit_line = results.fittedvalues
                residuals = vaz - fit_line
                
                # Plot the fitted line
                plt.figure(figsize=(12,8))
                plt.scatter(avgu,vaz,color='green')
                plt.plot(avgu, fit_line, color='gold', label='Weighted fitted line',linewidth=7)
                
                # Calculate the standard deviation of the residuals
                std_residuals = np.std(residuals)
                
                # Create error bands
                #plt.fill_between(avgu, fit_line - std_residuals, fit_line + std_residuals, color='red', alpha=0.3, label='Uncertainty region')
                plt.axvline(0,color='red',linewidth=5)
                # Add labels and legend
                plt.ylabel(r'Azimuth [$\circ$]',fontsize=20)
                plt.xlabel(r'$\bar{c_{U}}$', fontsize = 20)
                plt.xlim(-0.025, 0.025)
                #plt.ylim([50.5, 54.5])
                plt.yticks(fontsize=15)
                plt.xticks(fontsize=15)
                #plt.title('Weighted Linear Regression with Fit Error')
                plt.grid(True)
                #plt.legend(fontsize=20,loc='upper left')
                plt.show()
                
                # Print regression results
                print(results.summary())
                
                # Print regression results and specific parameters
                print("Slope:", results.params[1])
                print("Intercept:", results.params[0])
                print("R-squared:", results.rsquared)
                print("Standard Error of [Intercept, Slope]:", results.bse)
                az_slope_error = results.bse[1]
                azimuth_error = results.bse[0]
                az_slope = results.params[1]
                azimuth = results.params[0]
                
                print('Sun', saz, sza, 'NP', altitude, azimuth)
                
                save = input("Save neutral point estimation to file? (Yes/No): ")
                if save.lower() in ['yes', 'y']:
                    if 'Neutral Point Estimation' not in f:
                        np_est = f.create_group("Neutral Point Estimation")
                    else: 
                        del f['Neutral Point Estimation']
                        np_est = f.create_group("Neutral Point Estimation")

                    np_est.create_dataset('Estimation NP Location (alt,az) [deg]', data = np.array([altitude, azimuth]))
                    np_est.attrs['Zenith Error [arcseconds]'] = altitude_error * 3600
                    np_est.attrs['Azimuth Error [arcseconds]'] = azimuth_error * 3600
                    print("Neutral point estimation saved.")
                    f['Measurement_Metadata'].attrs['Processed Level'] = 'Level 3'
                else :
                    continue
                
        f['Measurement_Metadata'].attrs['Processed Level'] = 'Level 2'
        f.close()
        
    except Exception as e:
        print(f'Error opening file {files[idx]} — {e}')
        continue


    