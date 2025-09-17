# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:07:14 2024

@author: ULTRASIP_1
"""

import time
import sys
import threading
from typing import Optional, Tuple
import cv2
from vmbpy import *


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    if usage:
        print_usage()

    sys.exit(return_code)


def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]

#For continuous stream 
# def parse_args() -> Tuple[Optional[str], AllocationMode]:
#     args = sys.argv[1:]
#     argc = len(args)

#     allocation_mode = AllocationMode.AnnounceFrame
#     cam_id = ""
#     for arg in args:
#         if arg in ('/h', '-h'):
#             print_usage()
#             sys.exit(0)
#         elif arg in ('/x', '-x'):
#             allocation_mode = AllocationMode.AllocAndAnnounceFrame
#         elif not cam_id:
#             cam_id = arg

#     if argc > 2:
#         abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

#     return (cam_id if cam_id else None, allocation_mode)


def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)

            except VmbCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vmb.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')

            return cams[0]


def setup_camera(cam: Camera,exp_time): 
    with cam:
        try:

            cam.ExposureTime.set(exp_time)

        except (AttributeError, VmbFeatureError):
            pass

        # Enable white balancing if camera supports it
        try:
            cam.BalanceWhiteAuto.set('Off')

        except (AttributeError, VmbFeatureError):
            pass
        
        try:
            cam.GainAuto.set(False)

        except (AttributeError, VmbFeatureError):
            pass
        
        try: 
            cam.set_pixel_format(PixelFormat.Mono12)
        
        except (AttributeError, VmbFeatureError):
            pass

class Handler:
    pass
