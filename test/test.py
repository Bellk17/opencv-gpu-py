#!/usr/bin/env python

import os
import cv2
import cv2gpu
import datetime

this_dir = os.path.dirname(os.path.realpath(__file__))
image_file = os.path.join(this_dir, 'family.jpg')

cascade_file_cpu = os.path.join(this_dir, 'haarcascade_frontalface_default.xml')
cascade_file_gpu = os.path.join(this_dir, 'haarcascade_frontalface_default_cuda.xml')

def main():

    if cv2gpu.is_cuda_compatible():
        cv2gpu.init_gpu_detector(cascade_file_gpu)
    else:
        cv2gpu.init_cpu_detector(cascade_file_cpu)
    
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)

    dims = image.shape
    image = image.ravel()

    # Run once it initialize gpu with instructions
    faces = cv2gpu.find_faces(dims, image)

    start = datetime.datetime.now()
    faces = cv2gpu.find_faces(dims, image)

    image = cv2.imread(image_file)
    for (x, y, w, h) in faces:
        print('X: {x}, Y: {y}, width: {w}, height: {h}'.format(x=x, y=y, w=w, h=h))
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.imshow('faces', image)
        #cv2.waitKey(0)
    print('Time processed (s): ' + str(datetime.datetime.now() - start))

if __name__ == '__main__':
    main()
