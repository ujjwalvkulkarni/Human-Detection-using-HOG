# CSGY 6643 - Computer Vision Project 2 - Human Detection using HOG
# Completed by group of 2:
#   1. Harsh Sonthalia - hs4226
#   2. Ujjwal Vikram Kulkarni - uk2011

import HOG
import cv2
import glob
import os
import numpy as np 
from cv2 import imwrite

#setting paths for test image folders
positive_path_test = '/Users/ujjwalkulkarni/Desktop/Image Data/Test images (Pos)/'
negative_path_test = '/Users/ujjwalkulkarni/Desktop/Image Data/Test images (Neg)/'

#Creating directories to save the gradient images
os.makedirs(os.path.join(positive_path_test,"Gradient"))
os.makedirs(os.path.join(negative_path_test,"Gradient"))

#Saving the gradient images for positive dataset
for filename in glob.glob(os.path.join(positive_path_test,"*.bmp")):
    imageread = cv2.imread(filename)
    head, tail = os.path.split(filename)
    GradientPath = os.path.join(head,"Gradient",tail)
    Gradient_Magnitude, HOG_Vector = HOG.FaceDetection(imageread)
    imwrite(GradientPath,Gradient_Magnitude)

#Saving the gradient images for negative dataset
for filename in glob.glob(os.path.join(negative_path_test,"*.bmp")):
    imageread = cv2.imread(filename)
    head, tail = os.path.split(filename)
    GradientPath = os.path.join(head,"Gradient",tail)
    Gradient_Magnitude,HOG_Vector = HOG.FaceDetection(imageread)
    imwrite(GradientPath,Gradient_Magnitude)