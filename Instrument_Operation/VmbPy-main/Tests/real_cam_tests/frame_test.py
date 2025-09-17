"""BSD 2-Clause License

Copyright (c) 2022, Allied Vision Technologies GmbH
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import copy
import ctypes
import os
import sys

from vmbpy import *
from vmbpy.frame import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpers import VmbPyTestCase


class CamFrameTest(VmbPyTestCase):
    def setUp(self):
        self.vmb = VmbSystem.get_instance()
        self.vmb._startup()

        try:
            self.cam = self.vmb.get_camera_by_id(self.get_test_camera_id())

        except VmbCameraError as e:
            self.vmb._shutdown()
            raise Exception('Failed to lookup Camera.') from e

    def tearDown(self):
        self.vmb._shutdown()

    def test_verify_buffer(self):
        # Expectation: A Frame buffer shall have exactly the specified size on construction.
        # Allocation is performed by vmbpy
        self.assertEqual(Frame(0, AllocationMode.AnnounceFrame).get_buffer_size(), 0)
        self.assertEqual(Frame(1024, AllocationMode.AnnounceFrame).get_buffer_size(), 1024)
        self.assertEqual(Frame(1024 * 1024, AllocationMode.AnnounceFrame).get_buffer_size(),
                         1024 * 1024)

    def test_verify_no_copy_empty_buffer_access(self):
        # Expectation: Accessing the internal buffer must not create a copy
        # frame._buffer is only set on construction if buffer is allocated by vmbpy
        frame = Frame(10, AllocationMode.AnnounceFrame)
        self.assertEqual(id(frame._buffer), id(frame.get_buffer()))

    def test_verify_no_copy_filled_buffer_access(self):
        # Expectation: Accessing the internal buffer must not create a copy
        for allocation_mode in AllocationMode:
            with self.subTest(f'allocation_mode={str(allocation_mode)}'):
                with self.cam:
                    frame = self.cam.get_frame(allocation_mode=allocation_mode)
                self.assertEqual(id(frame._buffer), id(frame.get_buffer()))

    def test_get_id(self):
        # Expectation: get_id() must return None if Its locally constructed
        # else it must return the frame id.
        for allocation_mode in AllocationMode:
            with self.subTest(f'allocation_mode={str(allocation_mode)}'):
                self.assertIsNone(Frame(0, allocation_mode).get_id())

                with self.cam:
                    self.assertIsNotNone(
                        self.cam.get_frame(allocation_mode=allocation_mode).get_id())

    def test_get_timestamp(self):
        # Expectation: get_timestamp() must return None if Its locally constructed
        # else it must return the timestamp.
        for allocation_mode in AllocationMode:
            with self.subTest(f'allocation_mode={str(allocation_mode)}'):
                self.assertIsNone(Frame(0, allocation_mode).get_timestamp())

                with self.cam:
                    self.assertIsNotNone(
                        self.cam.get_frame(allocation_mode=allocation_mode).get_timestamp())

    def test_get_offset(self):
        # Expectation: get_offset_x() must return None if Its locally constructed
        # else it must return the offset as int. Same goes for get_offset_y()
        for allocation_mode in AllocationMode:
            with self.subTest(f'allocation_mode={str(allocation_mode)}'):
                self.assertIsNone(Frame(0, allocation_mode).get_offset_x())
                self.assertIsNone(Frame(0, allocation_mode).get_offset_y())

                with self.cam:
                    frame = self.cam.get_frame(allocation_mode=allocation_mode)
                    self.assertIsNotNone(frame.get_offset_x())
                    self.assertIsNotNone(frame.get_offset_y())

    def test_get_dimension(self):
        # Expectation: get_width() must return None if Its locally constructed
        # else it must return the offset as int. Same goes for get_height()
        for allocation_mode in AllocationMode:
            with self.subTest(f'allocation_mode={str(allocation_mode)}'):
                self.assertIsNone(Frame(0, allocation_mode).get_width())
                self.assertIsNone(Frame(0, allocation_mode).get_height())

                with self.cam:
                    frame = self.cam.get_frame(allocation_mode=allocation_mode)
                    self.assertIsNotNone(frame.get_width())
                    self.assertIsNotNone(frame.get_height())

    def test_deepcopy(self):
        # Expectation: a deepcopy must clone the frame buffer with its contents an
        # update the internally store pointer in VmbFrame struct.
        for allocation_mode in AllocationMode:
            with self.subTest(f'allocation_mode={str(allocation_mode)}'):
                with self.cam:
                    frame = self.cam.get_frame(allocation_mode=allocation_mode)

                frame_cpy = copy.deepcopy(frame)

                # Ensure frames and their members are not the same object
                self.assertNotEqual(id(frame), id(frame_cpy))
                self.assertNotEqual(id(frame._buffer), id(frame_cpy._buffer))
                self.assertNotEqual(id(frame._frame), id(frame_cpy._frame))

                # Ensure that both buffers have the same size and contain the same data.
                self.assertEqual(frame.get_buffer_size(), frame_cpy.get_buffer_size())
                self.assertTrue(all(a == b for a, b in zip(frame.get_buffer(),
                                                           frame_cpy.get_buffer())))

                # Ensure that internal Frame Pointer points to correct buffer.
                self.assertEqual(frame._frame.buffer,
                                 ctypes.cast(frame._buffer, ctypes.c_void_p).value)

                self.assertEqual(frame_cpy._frame.buffer,
                                 ctypes.cast(frame_cpy._buffer, ctypes.c_void_p).value)

                self.assertEqual(frame._frame.bufferSize, frame_cpy._frame.bufferSize)

    def test_get_pixel_format(self):
        # Expectation: Frames have an image format set after acquisition
        for allocation_mode in AllocationMode:
            with self.subTest(f'allocation_mode={str(allocation_mode)}'):
                with self.cam:
                    self.assertNotEqual(
                        self.cam.get_frame(allocation_mode=allocation_mode).get_pixel_format(), 0)

    def test_incompatible_formats_value_error(self):
        # Expectation: Conversion into incompatible formats must lead to an value error
        for allocation_mode in AllocationMode:
            with self.subTest(f'allocation_mode={str(allocation_mode)}'):
                with self.cam:
                    frame = self.cam.get_frame(allocation_mode=allocation_mode)

                current_fmt = frame.get_pixel_format()
                convertable_fmt = current_fmt.get_convertible_formats()

                for fmt in PixelFormat.__members__.values():
                    if (fmt != current_fmt) and (fmt not in convertable_fmt):
                        self.assertRaises(ValueError, frame.convert_pixel_format, fmt)

    def test_convert_to_all_given_formats(self):
        # Expectation: A Series of Frame, each acquired with a different Pixel format
        # Must be convertible to all formats the given format claims its convertible to without any
        # errors.

        test_frames = []

        with self.cam:
            initial_pixel_format = self.cam.get_pixel_format()
            for fmt in self.cam.get_pixel_formats():
                self.cam.set_pixel_format(fmt)

                frame = self.cam.get_frame()

                self.assertEqual(fmt, frame.get_pixel_format())
                test_frames.append(frame)
            self.cam.set_pixel_format(initial_pixel_format)

        for frame in test_frames:
            original_fmt = frame.get_pixel_format()
            for expected_fmt in frame.get_pixel_format().get_convertible_formats():
                with self.subTest(f'convert {repr(original_fmt)} to {repr(expected_fmt)}'):
                    transformed_frame = frame.convert_pixel_format(expected_fmt)

                    self.assertEqual(expected_fmt, transformed_frame.get_pixel_format())
                    self.assertEqual(original_fmt, frame.get_pixel_format())

                    # Ensure that width and height of frames are identical (if both formats can be
                    # represented as numpy arrays)
                    try:
                        original_shape = frame.as_numpy_ndarray().shape
                        transformed_shape = transformed_frame.as_numpy_ndarray().shape
                        self.assertTupleEqual(original_shape[0:2], transformed_shape[0:2])
                    except VmbFrameError:
                        # one of the pixel formats does not support representation as numpy array
                        self.skipTest(f'{repr(original_fmt)} or {repr(expected_fmt)} is not '
                                      'representable as numpy array')
                    except ImportError:
                        # Numpy is not available. Checking shape is not possible.
                        self.skipTest('Numpy not installed. Could not check frame shapes for '
                                      'equality')

    def test_numpy_arrays_can_be_accessed_after_frame_is_garbage_collected(self):
        # Expectation: A numpy array that was created from a VmbPy frame is valid even if the
        # original VmbPy frame has been deleted and the garbage collector cleaned it up. The
        # lifetime of the VmbPy frame's self._buffer must be tied to both, the frame and the numpy
        # array. Otherwise a segfault occurs (execution aborts immediately!)

        # WARNING: IF A SEGFAULT IS CAUSED, THIS WILL IMMEDIATELY HALT ALL EXECUTION OF THE RUNNING
        # PROCESS. THIS MEANS THAT THE TEST CASE WILL NOT REALLY REPORT A FAILURE, BUT SIMPLY EXIT
        # WITHOUT BEING MARKED AS PASSED. ALL SUBSEQUENT TEST CASES WILL ALSO NOT BE EXECUTED
        with self.cam:
            compatible_formats = intersect_pixel_formats(OPENCV_PIXEL_FORMATS,
                                                         self.cam.get_pixel_formats())
            if not compatible_formats:
                self.skipTest(f'Camera does not support a compatible format. Available formats '
                              f'from camera are: {self.cam.get_pixel_formats()}. Numpy compatible '
                              f'formats are {OPENCV_PIXEL_FORMATS}')
            if self.cam.get_pixel_format() not in compatible_formats:
                self.cam.set_pixel_format(compatible_formats[0])
            frame = self.cam.get_frame()
        try:
            np_array = frame.as_numpy_ndarray()
        except ImportError:
            self.skipTest('Numpy is not imported')
        del frame

        # Ensure that garbage collection has cleaned up the frame object
        import gc
        gc.collect()

        # Perform some calculation with numpy array to ensure that access is possible
        self.assertNoRaise(np_array.mean)
