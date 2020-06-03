from scipy import *
from scipy import signal
from PIL import Image
import cv2
import numpy
import math
import imageio

##############################################################
#               Modified Canny.py
##############################################################
#   
#   The original canny in the master brach has a major
#   flaw , in that it is actually only sobel edge detector
#   and not really canny edge detector. The new and improved
#   canny edge dectector will be merged into the master 
#   branch afer it has been fixed, revamped and thoroughly
#   improved.
#   



def normalize(image):

    mi = 255
    ma = 0

    new = numpy.zeros((len(image),len(image[0]),1),dtype="uint8")
    #cv2.imshow("Black",new)

    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > ma:
                ma = image[i][j]
            if image[i][j] < mi:
                mi = image[i][j]
            new[i][j] = image[i][j]
    #cv2.imshow("Copy",new)

    for i in range(len(image)):
        for j in range(len(image[i])):
            #new[i][j] = int((image[i][j]-mi)/(ma-mi) % 256)
            if image[i][j]<128:
                new[i][j] = 0
            else :
                new[i][j] = 255

    #cv2.imshow("Normalize 0 to 1",new)

    for i in range(len(image)):
        for j in range(len(image[i])):
            new[i][j] = int((image[i][j]-mi)*255/(ma-mi) % 256)

    return new


def non_max_suppression(img, D):
    #print(img.shape)
    M, N = img.shape
    Z = numpy.zeros((M,N), dtype=numpy.int32)
    angle = D * 180 / numpy.pi
    angle[angle < 0] += 180
    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

def dthreshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.1):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = numpy.zeros((M,N), dtype=numpy.float32)
    
    weak = numpy.float32(25)
    strong = numpy.float32(255)
    
    strong_i, strong_j = numpy.where(img >= highThreshold)
    zeros_i, zeros_j = numpy.where(img < lowThreshold)
    
    weak_i, weak_j = numpy.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

#def hysteresis(img, weak, strong=255):
def hysteresis(img):
    img, weak, strong = dthreshold(img)
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


# Locating the image. If the image is not same then change to relative address.
usedImage = '../../Images/test.jpg'

# Opening the image into an array
img = numpy.array(Image.open(usedImage).convert("L"))

# Kernel to perform gaussian blur
kernel = [[0, 1, 0],
           [1, 2, 1],
           [0, 1, 0]]

# CONVOLUTION 1
# Performing gaussain blur by performing convolution with gaussian kernel.
# I could not code the convolution so I got irritated and used a function for
# convolution instead.
gaussian = signal.convolve(img, kernel, mode='same')

# Print array just to check the output. Can uncomment if you want.
# print ('Im: Convolution 1')
# print (gaussian)

# Saving the array with the blurred image as a JPG image
imageio.imwrite('./Outputs/smooth.jpeg', gaussian)
# cv2.imshow('smooth.jpeg', gaussian) # This statement does not work btw

# Kernel for Sobel X (using horizontal transformation)
kernelX = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]

# Kernel for Sobel Y (using vertical transformation)
kernelY = [[-1, -2, -1],
           [0, 0, 0],
           [1, 2, 1]]

# Kernel for diagonal Kirsch transformation 1
kernelXY = [[0,1,2],
            [-1,0,1],
            [-2,-1,0]]

# Kernel for diagonal Kirsch transformation 2
kernelYX = [[-2,-1,0],
            [-1,0,1],
            [0,1,2]]

# CONVOLUTION 2
# Performing convolution over the smoothed image with all the generated kernels.

# Generate output array imX of horizontal convolution
imX = signal.convolve(gaussian, kernelX, mode='same').astype(numpy.float32)
# Generate output array imY of vertical convolution
imY = signal.convolve(gaussian, kernelY, mode='same').astype(numpy.float32)
# Generate output array imX of horizontal convolution
imXY = signal.convolve(gaussian, kernelXY, mode='same').astype(numpy.float32)
# Generate output array imY of vertical convolution
imYX = signal.convolve(gaussian, kernelYX, mode='same').astype(numpy.float32)

# Printing arrays to console just to check
# print ('Im X: Convolution 2')
# print (imX)
# print ('Im Y: Convolution 2')
# print (imY)
# print ('Im XY: Convolution 2')
# print (imXY)
# print ('Im YX: Convolution 2')
# print (imYX)

# Saving the arrays created as JPG images
imageio.imwrite('./Outputs/imX.jpeg', imX)
imageio.imwrite('./Outputs/imY.jpeg', imY)
imageio.imwrite('./Outputs/imXY.jpeg', imXY)
imageio.imwrite('./Outputs/imYX.jpeg', imYX)
'''cv2.imshow('imX.jpeg', imX)
cv2.imshow('imY.jpeg', imY)
cv2.imshow('imXY.jpeg', imXY)
cv2.imshow('imYX.jpeg', imYX)'''

# Combining all the horizontal and vertical gradient approximations 
# to create the final canny edge detected image
#imFinal = numpy.lib.scimath.sqrt(imX*imX + imY*imY + imXY*imXY + imYX*imYX)
imFinal = numpy.lib.scimath.sqrt(imX*imX + imY*imY).astype(numpy.float32)

# Printing the canny edge detected image array just to check
# print ('Im Final: Combining Gradient Approximations')
# print (imFinal)

# Saving the final canny edge detection image as a JPG image
imageio.imwrite('./Outputs/canny.jpeg', imFinal)
# cv2.imshow('canny.jpeg', imFinal)

fin = normalize(imFinal)

G = numpy.hypot(imX, imY)
G = G / G.max() * 255
theta = numpy.arctan2(imX, imY)

temp = non_max_suppression(imFinal, theta)
#temp = normalize(temp)
imageio.imwrite('./Outputs/temp.jpeg', temp)

last = hysteresis(temp)
imageio.imwrite('./Outputs/last.jpeg', last)

print ('Finished Canny edge detection')
