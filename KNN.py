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
import numpy as np


#setting paths for test image folders
positive_path_test = '/Users/ujjwalkulkarni/Desktop/Image Data/Test images (Pos)/'
negative_path_test = '/Users/ujjwalkulkarni/Desktop/Image Data/Test images (Neg)/'

#setting paths for train image folders
positive_path_train = '/Users/ujjwalkulkarni/Desktop/Image Data/Training images (Pos)/'
negative_path_train = '/Users/ujjwalkulkarni/Desktop/Image Data/Training images (Neg)/'

imagename = []

#array to store the HOG Vectors of training images
training_x = []

#array to store the label of training images
training_y = []

#Reading the images and calculating their HOG vectors for Positive Images
for filename in glob.glob(os.path.join(positive_path_train,"*.bmp")):
    imageread = cv2.imread(filename)
    head, tail = os.path.split(filename)
    #extracting just the file name and creating it into a list
    imagename.append(tail)
    #storing the gradient magnitude and HOG vector of each image
    Gradient_Magnitude, HOG_vector = HOG.FaceDetection(imageread)
    #Appending the HOG vectors of each image after it's calculated
    training_x.append(HOG_vector)
    #Creating a label set for each image. Since, this is a positive image set, we will label it 1
    training_y.append(1)

#Reading the images and calculating their HOG vectors for Negative Images
for filename in glob.glob(os.path.join(negative_path_train,"*.bmp")):
    imageread = cv2.imread(filename)
    #storing the gradient magnitude and HOG vector of each image
    Gradient_Magnitude,HOG_vector = HOG.FaceDetection(imageread)
    head, tail = os.path.split(filename)
    imagename.append(tail)
    #Appending the HOG vectors of each image after it's calculated
    training_x.append(HOG_vector)
    #Creating a label set for each image. Since, this is a negative image set, we will label it 0
    training_y.append(0)

#Converting the appended array into numpy array to perform opertaions easily
training_x = np.float64(training_x)
training_y = np.array(training_y)


#Computing the similarity distances for classification of the positive image dataset
for filename in glob.glob(os.path.join(positive_path_test,"*.bmp")):
    #reading the test image
    TestImage = cv2.imread(filename)
    #extracting the HOG Vector of test image
    Gradient_Magnitude, Test_HOG = HOG.FaceDetection(TestImage)
    similarity = []
    #Finding the similarity of Input Image with each of the Training Image
    for HOG_vector in training_x:
        #Finding the minimum at each location in the vector, then summing all the minimums
        numerator = np.sum(np.minimum(Test_HOG, HOG_vector))
        #Finding the sum of the training image vector
        denominator = np.sum(HOG_vector)
        #Calculating the similarity of each training image with input image
        histogram_similarity = numerator/denominator
        #Appening all the similarities to create a list 
        similarity.append(histogram_similarity)
    #A dictionary having key as the similarity and pointing to the filename with which the similarity is associated
    pathname = dict(zip(similarity, imagename))
    #A dictionary having key as the similarity and pointing to the label with which the similarity is associated
    label_dict = dict(zip(similarity, training_y))
    #Sorting the similarities in descending order to have most similar images at the top
    sorted_dict = sorted(label_dict.keys(), reverse=True)
    
    #displaying the top 3 Nearest Neighbours for the positive dataset
    n=0
    #Counter to check the number of humans and non human images 
    counter_human = 0
    counter_nohuman = 0
    head, tail = os.path.split(filename)
    print('\n',tail)
    for i in sorted_dict:
        if label_dict[i] == 1:
            str = 'Human'
            counter_human = counter_human + 1
        else:
            str = 'No Human'
            counter_nohuman = counter_nohuman + 1
        #Displaying the top 3 neares neighbors and their values
        print("File name for " ,n+1," NN: ", pathname[i]," and distance: ",i, " with label: ", str)
        n = n + 1
        if n == 3:
            break
    #Classifying whether the input image has human or not
    if counter_human > counter_nohuman:
        print("Human Detected in image file: ",tail)
    else:
        print("No Human Detected in image file: ",tail)

#Computing the similarity distances and classification of the negative image dataset
for filename in glob.glob(os.path.join(negative_path_test,"*.bmp")):
    #reading the test image
    TestImage = cv2.imread(filename)
    #extracting the HOG Vector of test image
    Gradient_Magnitude, Test_HOG = HOG.FaceDetection(TestImage)
    similarity = []
    #Finding the similarity of Input Image with each of the Training Image
    for HOG_vector in training_x:
        #Finding the minimum at each location in the vector, then summing all the minimums
        numerator = np.sum(np.minimum(Test_HOG, HOG_vector))
        #Finding the sum of the training image vector
        denominator = np.sum(HOG_vector)
        #Calculating the similarity of each training image with input image
        histogram_similarity = numerator/denominator
        #Appening all the similarities to create a list 
        similarity.append(histogram_similarity)
    #A dictionary having key as the similarity and pointing to the filename with which the similarity is associated
    pathname = dict(zip(similarity, imagename))
    #A dictionary having key as the similarity and pointing to the label with which the similarity is associated
    label_dict = dict(zip(similarity, training_y))
    #Sorting the similarities in descending order to have most similar images at the top
    sorted_dict = sorted(label_dict.keys(), reverse=True)
    
    #displaying the top 3 Nearest Neighbours for the negative dataset
    n=0
    #Counter to check the number of humans and non human images 
    counter_human = 0
    counter_nohuman = 0
    head, tail = os.path.split(filename)
    print('\n',tail)
    for i in sorted_dict:
        if label_dict[i] == 1:
            str = 'Human'
            counter_human = counter_human + 1
        else:
            str = 'No Human'
            counter_nohuman = counter_nohuman + 1
        #Displaying the top 3 neares neighbors and their values
        print("File name for " ,n+1," NN: ", pathname[i]," and distance: ",i, " with label: ", str)
        n = n + 1
        if n == 3:
            break
    #Classifying whether the input image has human or not
    if counter_human > counter_nohuman:
        print("Human Detected in image file: ",tail)
    else:
        print("No Human Detected in image file: ",tail)

