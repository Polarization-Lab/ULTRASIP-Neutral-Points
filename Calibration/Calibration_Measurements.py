# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 14:10:15 2025
@author: ULTRASIP_1
NUC: obtain series of flat field images at different exposures 
Malus: Obtain images at different LP angles and input S0 
"""
#Import libraries 
from zaber_motion.ascii import Connection
import matplotlib.pyplot as plt
from zaber_motion import Units
from datetime import datetime
import uv_cam_functions as uv
import numpy as np
import serial
import time
import h5py
import os

#--------Constants and Metadata--------------------#
uv_wavelength = '355 10 nm FWHM'

outpath = 'E:/Calibration/'
Calibration_Type = 'Malus'
#Set filename for measurement using date/time
dt = datetime.now()
date_time = str(dt)
date = date_time[0:10].replace('-','')
timestamp = date_time[11:19].replace(':','_')

filename = Calibration_Type+'_'+date+'_'+timestamp+'.h5'
datapath = os.path.join(outpath,'Data')
if not os.path.exists(datapath):
        os.makedirs(datapath)
filename=os.path.join(datapath,filename)
hdf5_file = h5py.File(filename,"w")
meas = hdf5_file.create_group("Measurement_Metadata")
meas.attrs['Timestamp'] = timestamp
meas.attrs['Date'] = date
meas.attrs['Calibration Type'] = Calibration_Type
meas.attrs['UV Bandpass'] = uv_wavelength

if Calibration_Type == 'NUC':
    
    uv_exposures = np.linspace(4e2, 1e6, 150)
    angles = [0,45,90,135]
    
    #Connect to Rotation Motor
    connection = Connection.open_serial_port("COM6")
    device_list = connection.detect_devices()
    device = device_list[0]
    axis = device.get_axis(1)
    if not axis.is_homed():
        axis.home()
    axis.settings.set('maxspeed',100, Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)

    for ang in angles: 
        axis.move_absolute(ang, Units.ANGLE_DEGREES) 
        meas_angle = axis.get_position(Units.ANGLE_DEGREES)
        uvimage_data= []
        
        cam_id = uv.parse_args()
        with uv.VmbSystem.get_instance():
            with uv.get_camera(cam_id) as uvcam:
                for uv_exp in uv_exposures:
                    #UV Cam
                    # setup general camera settings and the pixel format in which frames are recorded
                    uv.setup_camera(uvcam,uv_exp)
                    handler = uv.Handler()  # Placeholder handler

                    try:
                        time1 = time.time()
                        # Capture a frame
                        frame = uvcam.get_frame()
                        # Directly save the frame data without using get_buffer_data_numpy
                        data = np.frombuffer(frame.get_buffer(),dtype=np.uint16)
                        uvimage_data = np.append(uvimage_data,data)
                        time2 = time.time()

                        uvmeastime = (time2-time1)/60
                        #print('uvmeas(min) =',uvmeastime)
                    finally: 
                        print('Next')
        
        print('saving')
        uvimg = hdf5_file.create_group(f"P_{ang} Measurements")
        uvimg.attrs['Angle of Linear Polarizer'] = meas_angle
        uvimg.attrs['UV Image Capture Time'] = uvmeastime
        uvimg.create_dataset('UV Raw Images', data = uvimage_data)
        uvimg.create_dataset('Exposure Times', data = uv_exposures)
       
    connection.close()
    # Close the HDF5 file
    hdf5_file.close()


if Calibration_Type == 'Malus':
    
    uv_exp = 1e6 
    angles = np.r_[0:365:5]
    runs = 5
    gt_angle = '0'
    meas.attrs['Angle of Generator Linear Polarizer'] = gt_angle
    meas.attrs["Runs for each angle"] = runs
    meas.attrs['Sampled Angles'] = angles
    meas.attrs['UV Bandpass'] = uv_wavelength

    
    #Connect to Rotation Motor
    connection = Connection.open_serial_port("COM6")
    device_list = connection.detect_devices()
    device = device_list[0]
    axis = device.get_axis(1)
    if not axis.is_homed():
        axis.home()
    axis.settings.set('maxspeed',100, Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)
    
    #Set up UV camera settings 
    cam_id = uv.parse_args()
    with uv.VmbSystem.get_instance():
        with uv.get_camera(cam_id) as uvcam:
            uv.setup_camera(uvcam,uv_exp)
            handler = uv.Handler()  # Placeholder handler
            
            for ang in angles: 
                uvimage_data = []
                axis.move_absolute(ang, Units.ANGLE_DEGREES) 
                meas_angle = axis.get_position(Units.ANGLE_DEGREES)
                
                try:
                    time1 = time.time()
                    # Capture a frame
                    for idx in range(0,runs):
                        frame = uvcam.get_frame()
                        # Directly save the frame data without using get_buffer_data_numpy
                        data = np.frombuffer(frame.get_buffer(),dtype=np.uint16)
                        uvimage_data = np.append(uvimage_data,data)
                    time2 = time.time()
                    
                finally:
                    uvmeastime = time2-time1
                    print('saving')
                    
                    uvimg = hdf5_file.create_group(f"P_{ang} Measurements")
                    uvimg.attrs['Angle of Analyzer Linear Polarizer'] = meas_angle
                    uvimg.attrs['UV Image Capture Time'] = uvmeastime
                    uvimg.attrs['Exposure Time'] = uv_exp
                    uvimg.create_dataset('UV Raw Images', data = uvimage_data)
               
    connection.close()
    # Close the HDF5 file
    hdf5_file.close()
                
    
    

