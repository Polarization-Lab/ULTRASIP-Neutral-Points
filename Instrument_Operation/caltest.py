# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 14:30:26 2025

@author: ULTRASIP_1
"""
#Import Libraries 
import numpy as np

# Define W-matrix of ULTRASIP (rows = analyzer vectors P0, P90, P45, P135)
Wideal = 0.5 * np.array([[1, 1, 0],
                    [1, -1, 0],
                    [1, 0, 1],
                    [1, 0, -1]])

#W-matrix cal 
Stokes_ideal = np.array([[1,1,1,1],[1,-1,0,0],[0,0,1,-1]])
vert_P0 = 0.25*np.ones((10,10))
vert_P45 = 0.25*np.ones((10,10))
vert_P90 = 0.25*np.ones((10,10))
vert_P135 = 0.25*np.ones((10,10))

hor_P0 = 0.5*np.ones((10,10))
hor_P45 =  0.5*np.ones((10,10))
hor_P90 =  0.5*np.ones((10,10))
hor_P135 =  0.5*np.ones((10,10))

forty_P0 =  0.75*np.ones((10,10))
forty_P45 =  0.75*np.ones((10,10))
forty_P90 =  0.75*np.ones((10,10))
forty_P135 =  0.75*np.ones((10,10))

thirty_P0 = np.ones((10,10))
thirty_P45 = np.ones((10,10))
thirty_P90 = np.ones((10,10))
thirty_P135 = np.ones((10,10))



flux_matrix = np.array([[hor_P0,vert_P0,forty_P0,thirty_P0],
                       [hor_P90,vert_P90,forty_P90,thirty_P90],
                       [hor_P45,vert_P45,forty_P45,thirty_P45],
                       [hor_P135,vert_P135,forty_P135,thirty_P135]]).T

W_meas = 0.5*( flux_matrix.reshape(4,4,10*10).T@np.linalg.pinv(Stokes_ideal))



