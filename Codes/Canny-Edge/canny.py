from scipy import *
from scipy import signal
from PIL import Image
import cv2
import numpy
import math
import imageio

# Locating the image. If the image is not same then change to relative address.
usedImage = '../../Images/2.jpg'

# Opening the image into an array
img = array(Image.open(usedImage).convert("L"))

# Kernel to perform gaussian blur
kernel = [[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]]

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
imX = signal.convolve(gaussian, kernelX, mode='same')
# Generate output array imY of vertical convolution
imY = signal.convolve(gaussian, kernelY, mode='same')
# Generate output array imX of horizontal convolution
imXY = signal.convolve(gaussian, kernelXY, mode='same')
# Generate output array imY of vertical convolution
imYX = signal.convolve(gaussian, kernelYX, mode='same')

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
imageio.imwrite('./Outputs/imX.jpeg', imX)
imageio.imwrite('./Outputs/imXY.jpeg', imXY)
imageio.imwrite('./Outputs/imYX.jpeg', imYX)
'''cv2.imshow('imX.jpeg', imX)
cv2.imshow('imY.jpeg', imY)
cv2.imshow('imXY.jpeg', imXY)
cv2.imshow('imYX.jpeg', imYX)'''

# Combining all the horizontal and vertical gradient approximations 
# to create the final canny edge detected image
imFinal = sqrt(imX*imX + imY*imY + imXY*imXY + imYX*imYX)

# Printing the canny edge detected image array just to check
# print ('Im Final: Combining Gradient Approximations')
# print (imFinal)

# Saving the final canny edge detection image as a JPG image
imageio.imwrite('./Outputs/canny.jpeg', imFinal)
# cv2.imshow('canny.jpeg', imFinal)

print ('Finished Canny edge detection')
