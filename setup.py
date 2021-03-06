#!/usr/bin/env python

import os
from distutils.core import setup, Extension
from subprocess import Popen, PIPE

this_dir = os.path.dirname(os.path.realpath(__file__))

cv2gpumodule = Extension('cv2gpu',
    define_macros = [('MAJOR_VERSION', '1'), ('MINOR_VERSION', '0')],
    sources = [os.path.join('src', 'face_detector.cpp'), os.path.join('src', 'cv2gpu.cpp')],
    include_dirs = ['/opencv/usr/include/opencv', '/opencv/usr/include/opencv2', '/opencv/usr/include'],
    libraries = ["opencv_cudacodec" ,"opencv_imgcodecs" ,"opencv_shape" ,"opencv_cudabgsegm" ,"opencv_videostab" ,"opencv_calib3d" ,"opencv_cudaarithm" ,"opencv_cudawarping" ,"opencv_cudafilters" ,"opencv_photo" ,"opencv_video" ,"opencv_ml" ,"opencv_features2d" ,"opencv_cudastereo" ,"opencv_cudalegacy" ,"opencv_cudev" ,"opencv_dnn" ,"opencv_imgproc" ,"opencv_cudaobjdetect" ,"opencv_videoio" ,"opencv_flann" ,"opencv_cudaimgproc" ,"opencv_core" ,"opencv_superres" ,"opencv_stitching" ,"opencv_cudaoptflow" ,"opencv_highgui" ,"opencv_objdetect" ,"opencv_cudafeatures2d"],
    library_dirs = ['/opencv/usr/lib'],
    extra_compile_args = ['-std=c++11'])

setup (name = 'cv2gpu',
    version = '1.0',
    description = 'OpenCV GPU Bindings',
    author = 'Alexander Koumis and Matthew Carlis',
    author_email = 'alexander.koumis@sjsu.edu, matthew.carlis@sjsu.edu',
    url = 'https://docs.python.org/extending/building',
    long_description = '''
    OpenCV GPU Bindings
    ''',
    ext_modules = [cv2gpumodule])
