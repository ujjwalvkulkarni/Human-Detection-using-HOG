# CSGY 6643 - Computer Vision Project 2 - Human Detection using HOG
# Completed by group of 2:
#   1. Harsh Sonthalia - hs4226
#   2. Ujjwal Vikram Kulkarni - uk2011
    
import math
import os
import cv2
import numpy as np
from cv2 import imwrite

#computing gradient magnitude and gradient angle using Prewitt's operator 
def Prewitt(gray_image):

    #initialising the Prewitt masks
    prewittX = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])

    prewittY = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])
    
    height = gray_image.shape[0]
    width = gray_image.shape[1]
    
    # We define two 2D arrays for the horizontal and vertical gradient
    horizontal_grad = np.empty((height, width))
    vertical_grad = np.empty((height, width))
    
    # Calculating the Vertical Gradient of Prewitt filter 
    for row in range(height):
        for col in range(width):
            #For boundary pixels, setting the value as 0
            if 0 <= row < 1 or height - 1 <= row <= height - 1 or 0 <= col < 1 or width - 1 <= col <= width - 1:
                vertical_grad[row][col] = 0
                
            else:
                #Sequentially adding the values to calculate vertical gradient
                sum = 0
                for i in range(3):
                    for j in range(3):
                        sum = sum + (gray_image[row - 1 + i][col - 1 + j]) * prewittX[i][j]
                        
                vertical_grad[row][col] = sum 

    # Calculating the Horizontal Gradient of Prewitt filter            
    for row in range(height):
        for col in range(width):
            #For boundary pixels, setting the value as 0
            if 0 <= row < 1 or height - 1 <= row <= height - 1 or 0 <= col < 1 or width - 1 <= col <= width - 1:
                horizontal_grad[row][col] = 0
                
            else:
                #Sequentially adding the values to calculate horizontal gradient
                sum = 0
                for i in range(3):
                    for j in range(3):
                        sum = sum + (gray_image[row - 1 + i][col - 1 + j]) * prewittY[i][j]
                        
                horizontal_grad[row][col] = sum
    #Taking absolute value to remove negative values and normalising the gradient values by a factor of 3              
    horizontal_grad_normal = np.true_divide(np.abs(horizontal_grad), 3)
    vertical_grad_normal = np.true_divide(np.abs(vertical_grad), 3)

    # We create a third 2D array to calculate the Gradient Magnitude
    grad_magnitude = np.empty((height, width))

    # Grad(x,y) = square root(x^2 + y^2), where x and y are the Horizontal and Vertical Gradients respectively
    for k in range(height):
        for l in range(width):
            if horizontal_grad_normal[k][l] == 0 and vertical_grad_normal[k][l] == 0:
              grad_magnitude[k][l] = 0
            else:  
              grad_magnitude[k][l] = math.sqrt((horizontal_grad_normal[k][l]*horizontal_grad_normal[k][l]) + (vertical_grad_normal[k][l]*vertical_grad_normal[k][l]))
    #Rounding off and Normalizing the Gradient Magnitude
    grad_magnitude = np.round(grad_magnitude)
    grad_magnitude = np.divide(grad_magnitude,2)

    # We define a fourth 2D array to calculate the gradient angle
    grad_angle = np.empty((height, width))
    
    # Angle = inverseTan(y/x), where x and y are the Horizontal and Vertical Gradients respectively
    for k in range(height):
        for l in range(width):
            # Condition to avoid Divison by 0 error
            if horizontal_grad_normal[k][l] == 0 or vertical_grad_normal[k][l] == 0:
                grad_angle[k][l] = 0
                
            else:    
                grad_angle[k][l] = math.degrees( math.atan(horizontal_grad[k][l]/vertical_grad[k][l]))

    #Converting all negative angles to positive while maintaining the range between 0 and 180
    for k in range(height):
        for l in range(width):
            if grad_angle[k][l] < 0:
                grad_angle[k][l] = grad_angle[k][l] + 180
    return grad_magnitude, grad_angle


#Computing Histogram of oriented gradients
def HOG(prewitt_output, gradientAngle):
    height = prewitt_output.shape[0]
    width = prewitt_output.shape[1]
    HOGvect = np.array([])
    TotalVect = np.array([])
    normalized_HOG = np.array([])

    #Calculating the HOG for the whole image 
    for m in range(0, height-8, 8):
        for n in range(0, width-8, 8):
            HOGvect = np.array([])
            temp = np.array([])

            #The loop for calculating HOG of each block
            for i in range(m, 16+m, 8):
                for j in range(n, 16+n, 8):
                    bin = np.zeros(9)

                    #The loop for calculating HOG of each cell
                    for k in range(i, 8+i):
                        for l in range(j, 8+j):

                            #Binning the magnitude values based on the gradient angle 
                            if gradientAngle[k][l] >= 10 and gradientAngle[k][l] <= 30:
                                rat1 = (30 - gradientAngle[k][l])/20
                                bin[0] = bin[0] + rat1*prewitt_output[k][l]
                                rat2 = (gradientAngle[k][l] - 10)/20
                                bin[1] = bin[1] + rat2*prewitt_output[k][l]

                            elif gradientAngle[k][l] > 30 and gradientAngle[k][l] <= 50:
                                rat1 = (50 - gradientAngle[k][l])/20
                                bin[1] = bin[1] + rat1*prewitt_output[k][l]
                                rat2 = (gradientAngle[k][l] - 30)/20
                                bin[2] = bin[2] + rat2*prewitt_output[k][l]

                            elif gradientAngle[k][l] > 50 and gradientAngle[k][l] <= 70:
                                rat1 = (70 - gradientAngle[k][l])/20
                                bin[2] = bin[2] + rat1*prewitt_output[k][l]
                                rat2 = (gradientAngle[k][l] - 50)/20
                                bin[3] = bin[3] + rat2*prewitt_output[k][l]

                            elif gradientAngle[k][l] > 70 and gradientAngle[k][l] <= 90:
                                rat1 = (90 - gradientAngle[k][l])/20
                                bin[3] = bin[3] + rat1*prewitt_output[k][l]
                                rat2 = (gradientAngle[k][l] - 70)/20
                                bin[4] = bin[4] + rat2*prewitt_output[k][l]

                            elif gradientAngle[k][l] > 90 and gradientAngle[k][l] <= 110:
                                rat1 = (110 - gradientAngle[k][l])/20
                                bin[4] = bin[4] + rat1*prewitt_output[k][l]
                                rat2 = (gradientAngle[k][l] - 90)/20
                                bin[5] = bin[5] + rat2*prewitt_output[k][l]

                            elif gradientAngle[k][l] > 110 and gradientAngle[k][l] <= 130:
                                rat1 = (130 - gradientAngle[k][l])/20
                                bin[5] = bin[5] + rat1*prewitt_output[k][l]
                                rat2 = (gradientAngle[k][l] - 110)/20
                                bin[6] = bin[6] + rat2*prewitt_output[k][l] 

                            elif gradientAngle[k][l] > 130 and gradientAngle[k][l] <= 150:
                                rat1 = (150 - gradientAngle[k][l])/20
                                bin[6] = bin[6] + rat1*prewitt_output[k][l]
                                rat2 = (gradientAngle[k][l] - 130)/20
                                bin[7] = bin[7] + rat2*prewitt_output[k][l]

                            elif gradientAngle[k][l] > 150 and gradientAngle[k][l] <= 170:
                                rat1 = (170 - gradientAngle[k][l])/20
                                bin[7] = bin[7] + rat1*prewitt_output[k][l]
                                rat2 = (gradientAngle[k][l] - 150)/20
                                bin[8] = bin[8] + rat2*prewitt_output[k][l]

                            elif gradientAngle[k][l] >= 0  and gradientAngle[k][l] < 10:
                                rat1 = (10 - gradientAngle[k][l])/20
                                bin[8] = bin[8] + rat1*prewitt_output[k][l]
                                rat2 = (20 - (10 - gradientAngle[k][l]))/20
                                bin[0] = bin[0] + rat2*prewitt_output[k][l]

                            elif gradientAngle[k][l] > 170  and gradientAngle[k][l] <= 180:
                                rat1 = (180 - gradientAngle[k][l])/20
                                bin[8] = bin[8] + rat1*prewitt_output[k][l]
                                rat2 = (20 - (180 - gradientAngle[k][l]))/20
                                bin[0] = bin[0] + rat2*prewitt_output[k][l] 
   
                    #concatenating the HOG of cells for each block
                    HOGvect = np.concatenate((HOGvect,bin))

            #normalizing each block us L2 normalization         
            block = np.sum(np.square(HOGvect))
            sq = math.sqrt(block)
            #condition to prevent NAN values
            if sq == 0 or block == 0:
                sq = 1
            normalized_HOG=  np.divide(HOGvect,sq)

            #concatenating the HOG of normalized blocks the image
            TotalVect = np.concatenate((TotalVect,normalized_HOG))         
    return TotalVect

#the driver funtion to retrieve all the output values
def FaceDetection(image):
    #extraction of each channel
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    #converting the image into a grayscale image
    gray = np.round(0.299 * r + 0.587 * g + 0.114 * b)
    prewitt_output, gradientAngle = Prewitt(gray)
    HOG_output = HOG(prewitt_output, gradientAngle)
    return prewitt_output, HOG_output
