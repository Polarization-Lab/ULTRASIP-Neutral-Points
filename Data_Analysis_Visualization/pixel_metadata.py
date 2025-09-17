# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:33:35 2024

@author: ULTRASIP_1
"""

import numpy as np

    

def pixel_geometry(HFOV,VFOV,img_x,img_y,x_center, y_center, Pan, Tilt, Sun_Position_Azimuth, Sun_Position_Altitude):
    
        view_az = np.zeros((img_x,img_y))
        view_zen = np.zeros((img_x,img_y))

        # Create the grid of indices
        x = np.arange(img_x)
        y = np.arange(img_y)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        # Compute deviations
        x_dev = (x_center - xx) * HFOV
        y_dev = (y_center - yy) * VFOV

        # Compute view angles
        view_az = np.radians(Pan - y_dev)
        view_zen = np.radians(Tilt - x_dev)
        sun_az = np.radians(Sun_Position_Azimuth)
        sun_zen = np.radians(Sun_Position_Altitude)
        
        return view_az, view_zen, sun_az, sun_zen