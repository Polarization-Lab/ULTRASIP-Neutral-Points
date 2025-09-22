# -*- coding: utf-8 -*-
"""
@author: C. M. DeLeon
ULTRASIP User Manual: 
    https://www.overleaf.com/read/hkkghcvgrrdt#c1c7d8

Code Description: 
    This code performs a sequence of "stop-and-stare" measurements relative to the sun.
    Such that polarization measurements are taken within the solar principal plane to find the neurtal point(s).
    
    
Pseudocode: 
"""

#Import libraries 
import numpy as np
import serial
import time
import h5py,pytz
import os
import matplotlib.pyplot as plt
from zaber_motion import Units
from zaber_motion.ascii import Connection
from datetime import datetime
from suncalc import get_position
import moog_functions as mf
import uv_cam_functions as uv
#import vis_cam_functions as vis

#----------------Constants and Metadata: CHANGE AS NEEDED----------------#
uv_wavelength = '355 FWHM 10nm'
#vis_wavelengths = '470,525,635 nm; Bayer Fitler'
aq_num = -1 #index for later
angles = [0,45,90,135]
#angles = [135,90,45,0]
Location = 'NormRoof'
#Get from Garmin GPS
latitude = 45.66487 #32.23134;
longitude = -111.04800 #-110.94712;

#Offsets needed -- from homing procedure (subtracted from calculated sun position)
tilt_offset = 0
pan_offset = 1.5 #to set origin to sun azimuth
#Scan range from sun, 0 is sun
start_tilt = 0
end_tilt = 4
step_tilt = 2

#Exposures
uv_exp_initial = 8e2
uv_exp_mid =  1e4
uv_exp_max = 1e5

#Data main directory 
outpath = 'D:/Data'

#Make new folder for today's date to save the data (if it doesn't exist)
#Set filename for measurement using date/time
dt = datetime.now()
date_time = str(dt)
date = date_time[0:10].replace('-','')
timestamp = date_time[11:19].replace(':','_')
filename = Location+'_'+date+'_'+timestamp+'.h5'
datapath = os.path.join(outpath,str(date_time[0:10].replace('-','_')))
if not os.path.exists(datapath):
    os.makedirs(datapath)
filename=os.path.join(datapath,filename)

# ----------------------------#Connect to motors#-------------------------#
# #Connect to Moog
# #Configure port connection
moog = serial.Serial()
moog.baudrate = 9600
moog.port = 'COM2'
moog.open()
mf.init_autobaud(moog);
mf.get_status_jog(moog)

#Connect to Rotation Motor
connection = Connection.open_serial_port("COM6")
device_list = connection.detect_devices()
device = device_list[0]
axis = device.get_axis(1)
if not axis.is_homed():
      axis.home()
axis.settings.set('maxspeed',100, Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)
#NOTE: Connection to DoFP and UV Cam done inline during image acqusition

#Desired Structure 
#File Metadata: Date, Time, Lat, Long
#Group_# = Aquisition_#
#Aquisition_#_Metadata: Time, Pan/Tilt, Sun Position, UV_Exposure, DoFP_Exposure
#Aquisition_Dataset(s): UV_ImageData, DoFP_ImageData

hdf5_file = h5py.File(filename,"w")
meas = hdf5_file.create_group("Measurement_Metadata")
meas.attrs['Latitude'] = str(latitude)
meas.attrs['Longitude'] = str(longitude)
meas.attrs['Pan_Offset'] = str(pan_offset)
meas.attrs['Tilt_Offset'] = str(tilt_offset)

#Get sun position and set to initial pan and tilt value for moog
dt = datetime.now()
sun_pos = get_position(dt, longitude, latitude)
#Set initial position based on sun location
pan = np.degrees(sun_pos['azimuth'])
tilt = np.degrees(sun_pos['altitude']) 

#Move Moog to Sun Position
mf.mv_to_coord(moog,int((pan-pan_offset)*10),int((tilt-tilt_offset)*10))
mf.get_status_jog(moog)
time.sleep(1)

measstart=time.time()

for dtilt in range(start_tilt,end_tilt,step_tilt):
    
    aq_num = aq_num + 1
    
    if dtilt == 0:
        uv_exp = uv_exp_initial
        
    if dtilt > 4 or dtilt < 0:
        uv_exp = uv_exp_mid
        
    if dtilt > 12:
        uv_exp = uv_exp_max
        
    mf.get_status_jog(moog)
    
    #Get sun position and set to initial pan and tilt value for moog
    dt = datetime.now()
    sun_pos = get_position(dt, longitude, latitude)
    
    #Set initial position based on sun location
    pan = np.degrees(sun_pos['azimuth']) 
    tilt = np.degrees(sun_pos['altitude']) + dtilt
    
    mf.mv_to_coord(moog,int((pan- pan_offset)*10),int((tilt-tilt_offset)*10)) 
    time.sleep(0.1)
    
    mf.get_status_jog(moog)
    #time.sleep(0.1)
    
    uvimage_data=[]
    cam_id = uv.parse_args()
    with uv.VmbSystem.get_instance():
        with uv.get_camera(cam_id) as uvcam:
            # setup general camera settings and the pixel format in which frames are recorded
            uv.setup_camera(uvcam,uv_exp)
            handler = uv.Handler()  # Placeholder handler

            try:
                date_time = str(dt)
                timestamp = date_time[11:19].replace(':','_')
                time1 = time.time()
                for angle in angles:
                    axis.move_absolute(angle, Units.ANGLE_DEGREES)  
                    #print(axis.get_position(Units.ANGLE_DEGREES))
                    axis.wait_until_idle()
                    time.sleep(1)
                    # Capture a frame
                    frame = uvcam.get_frame()
                    # Directly save the frame data without using get_buffer_data_numpy
                    data = np.frombuffer(frame.get_buffer(),dtype=np.uint16)
                    uvimage_data = np.append(uvimage_data,data)
                time2 = time.time()
                uvmeastime = (time2-time1)
                print('uvmeas =',uvmeastime)
                axis.home()
            finally: 
                print('Saving')
        
    aq = hdf5_file.create_group(f"Aquistion_{aq_num}")
    aq.attrs['Timestamp MDT'] = timestamp
    utc_time = str(dt.astimezone(pytz.utc))
    utc_timestamp = utc_time[11:19].replace(':','_')
    aq.attrs['Timestamp UTC'] = utc_timestamp
    aq.attrs['Pan'] = pan
    aq.attrs['Tilt'] = tilt
    aq.attrs['Sun Position Azimuth'] = np.degrees(sun_pos['azimuth'])
    aq.attrs['Sun Position Altitude'] = np.degrees(sun_pos['altitude'])


    uvimg = aq.create_group('UV Image Data')
    uvimg.create_dataset('UV Raw Images', data = uvimage_data)
    uvimg.attrs['UV Exposure Time'] = uv_exp
    uvimg.attrs['UV Bandpass'] = uv_wavelength
    uvimg.attrs['UV Image Capture Time'] = uvmeastime
    uvimg.attrs['UV Polarizer Angles'] = str(angles)



measend=time.time()
print('Measurment Completed',((measend-measstart)))
meas.attrs['Total Measurement Time'] = ((measend-measstart)/60)

sun_pos = get_position(dt, longitude, latitude)
#Set initial position based on sun location
pan = np.degrees(sun_pos['azimuth']) 
tilt = np.degrees(sun_pos['altitude'])

#Home and disconnect everything
mf.get_status_jog(moog)
mf.mv_to_coord(moog,int((pan- pan_offset)*10),int(tilt*10)) 
time.sleep(4)
axis.home()
moog.close()
connection.close()
# Close the HDF5 file
hdf5_file.close()