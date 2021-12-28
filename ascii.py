# CSGY 6643 - Computer Vision Project 2 - Human Detection using HOG
# Completed by group of 2:
#   1. Harsh Sonthalia - hs4226
#   2. Ujjwal Vikram Kulkarni - uk2011

import HOG
import cv2
import os
import numpy as np 

#setting paths for test image folders
positive_path_test = '/Users/ujjwalkulkarni/Desktop/Image Data/Test images (Pos)/'
negative_path_test = '/Users/ujjwalkulkarni/Desktop/Image Data/Test images (Neg)/'

#setting paths for train image folders
positive_path_train = '/Users/ujjwalkulkarni/Desktop/Image Data/Training images (Pos)/'
negative_path_train = '/Users/ujjwalkulkarni/Desktop/Image Data/Training images (Neg)/'

#Saving the HOG Vector for crop001028a.bmp in a .txt file
filename = os.path.join(positive_path_train,"crop001028a.bmp")
imageread = cv2.imread(filename)
Gradient_Magnitude, HOG_Vector = HOG.FaceDetection(imageread)
save = np.savetxt('crop001028a.txt', HOG_Vector, newline='\n')   

#Saving the HOG Vector for crop001030c.bmp in a .txt file
filename = os.path.join(positive_path_train,"crop001030c.bmp")
imageread = cv2.imread(filename)
Gradient_Magnitude, HOG_Vector = HOG.FaceDetection(imageread)
save = np.savetxt('crop001030c.txt', HOG_Vector, newline='\n')   

#Saving the HOG Vector for 00000091a_cut.bmp in a .txt file
filename = os.path.join(negative_path_train,"00000091a_cut.bmp")
imageread = cv2.imread(filename)
Gradient_Magnitude, HOG_Vector = HOG.FaceDetection(imageread)
save = np.savetxt('00000091a_cut.txt', HOG_Vector, newline='\n')   

#Saving the HOG Vector for crop001278a.bmp in a .txt file
filename = os.path.join(positive_path_test,"crop001278a.bmp")
imageread = cv2.imread(filename)
Gradient_Magnitude, HOG_Vector = HOG.FaceDetection(imageread)
save = np.savetxt('crop001278a.txt', HOG_Vector, newline='\n')   

#Saving the HOG Vector for crop001500b.bmp in a .txt file
filename = os.path.join(positive_path_test,"crop001500b.bmp")
imageread = cv2.imread(filename)
Gradient_Magnitude, HOG_Vector = HOG.FaceDetection(imageread)
save = np.savetxt('crop001500b.txt', HOG_Vector, newline='\n')   

#Saving the HOG Vector for 00000090a_cut.bmp in a .txt file
filename = os.path.join(negative_path_test,"00000090a_cut.bmp")
imageread = cv2.imread(filename)
Gradient_Magnitude, HOG_Vector = HOG.FaceDetection(imageread)
save = np.savetxt('00000090a_cut.txt', HOG_Vector, newline='\n')   