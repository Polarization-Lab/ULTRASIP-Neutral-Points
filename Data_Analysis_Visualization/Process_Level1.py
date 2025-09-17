
"""
Created on Sun Jun 29 17:04:25 2025

@author: C.M.DeLeon
Run Process Level 0 first to get corrected polarization data products. 
Process Level 1: Pixel Geometry
"""

#Import Libraries
from pixel_metadata import pixel_geometry as pg
import matplotlib.pyplot as plt
import numpy as np
import h5py 
import glob
import os,cv2
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import time
#UV Lens Attributes- FOV/pixel degrees
HFOV =  0.0020
VFOV = HFOV

img_x = 2848
img_y = 2848


#Set Date of Measurements 
date = '2025_06_10'
start_time = time.time()
#Datapath
basepath = 'D:/Data'
folderdate = os.path.join(basepath,date)
files = glob.glob(f'{folderdate}/NormRoof_*.h5')
#files = glob.glob(f'{folderdate}/NormRoof*09_16_36*.h5')

idx = len(files) 
idx_array = np.arange(0,idx)

def process1(idx):
    print(f'Processing file {idx}: {files[idx]}')
    
    try:
        f = h5py.File(files[idx], 'r+')
        
        for aqnum in range(0, len(f.keys()) - 1):
            try:
                aq = f[f'Aquistion_{aqnum}']
                uv_data = aq['UV Image Data']
                
                Pan = aq.attrs['Pan'] 
                Tilt = aq.attrs['Tilt']
                                
                Sun_Position_Azimuth = aq.attrs['Sun Position Azimuth']
                Sun_Position_Altitude = aq.attrs['Sun Position Altitude']
                    
                if aqnum == 0: 
                    I = uv_data['I_corrected'][:]
                    
                    sza0 = aq.attrs['Sun Position Altitude']
                    
                    # plt.figure()
                    # plt.imshow(I,interpolation = 'None',cmap='gray')
                    # plt.colorbar()

                    # Apply a manual threshold (e.g., 90% of max intensity)
                    thresh_val = 0.95 * np.max(I.flatten())
                    _, thresh = cv2.threshold(I, thresh_val, 1, cv2.THRESH_BINARY)
                    
                    thresh_uint8 = (thresh * 255).astype(np.uint8)
                    
                    # Compute moments
                    M = cv2.moments(thresh_uint8)

                    # Calculate centroid
                    if M["m00"] != 0:
                        x_center = int(M["m10"] / M["m00"])
                        y_center = int(M["m01"] / M["m00"])

                       # print(f"Sun center: (x={x_center}, y={y_center})")
                    else:
                        print("Could not detect the sun.")
                        x_center = img_x/2
                        y_center = img_y/2

                    # plt.imshow(I, interpolation = 'None', cmap='gray')
                    # plt.scatter([x_center], [y_center], color='red')
                    # plt.title(f"Sun Center: ({x_center}, {y_center})")
                    # plt.show()
                    f['Measurement_Metadata'].attrs['Center Pixel (x,y)'] = np.array([x_center,y_center])
                    #Calculate geometry 
                    view_az, view_zen, sun_az, sun_zen = pg(sza0,HFOV,VFOV,img_x,img_y,x_center, y_center, Pan, Tilt, Sun_Position_Azimuth, Sun_Position_Altitude)
                
                    if 'view_az' not in uv_data:
                        uv_data.create_dataset('view_az', data = np.degrees(view_az))
                        uv_data.create_dataset('view_zen', data = np.degrees(view_zen))
                        uv_data.create_dataset('sun_az', data = np.degrees(sun_az))
                        uv_data.create_dataset('sun_zen', data = np.degrees(sun_zen))
                    elif 'view_az' in uv_data:
                        print('here')
                        del uv_data['view_az']
                        del uv_data['view_zen']
                        del uv_data['sun_az']
                        del uv_data['sun_zen']
                        uv_data.create_dataset('view_az', data = np.degrees(view_az))
                        uv_data.create_dataset('view_zen', data = np.degrees(view_zen))
                        uv_data.create_dataset('sun_az', data = np.degrees(sun_az))
                        uv_data.create_dataset('sun_zen', data = np.degrees(sun_zen))

                else: 

                    #Calculate geometry 
                    view_az, view_zen, sun_az, sun_zen = pg(sza0,HFOV,VFOV,img_x,img_y,x_center, y_center, Pan, Tilt, Sun_Position_Azimuth, Sun_Position_Altitude)
                
                    if 'view_az' not in uv_data:
                        uv_data.create_dataset('view_az', data = np.degrees(view_az))
                        uv_data.create_dataset('view_zen', data = np.degrees(view_zen))
                        uv_data.create_dataset('sun_az', data = np.degrees(sun_az))
                        uv_data.create_dataset('sun_zen', data = np.degrees(sun_zen))
                    elif 'view_az' in uv_data:
                        print('here')
                        del uv_data['view_az']
                        del uv_data['view_zen']
                        del uv_data['sun_az']
                        del uv_data['sun_zen']
                        uv_data.create_dataset('view_az', data = np.degrees(view_az))
                        uv_data.create_dataset('view_zen', data = np.degrees(view_zen))
                        uv_data.create_dataset('sun_az', data = np.degrees(sun_az))
                        uv_data.create_dataset('sun_zen', data = np.degrees(sun_zen))

            except KeyError as e:
                print(f'Skipping Acquisition {aqnum} in file {idx} — {e}')
                continue
    finally: 
        #print('done')
        f['Measurement_Metadata'].attrs['Processed Level'] = 'Level 1'
        f.close()
        
threads = cpu_count()

with ThreadPool(threads) as p:
    p.map(process1, idx_array)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")