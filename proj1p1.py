import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

#example command:
#python proj1p1.py ../Images/HandsMed/Hand_0000003.jpg HOG true

if len(sys.argv) != 4:
	print("Error: This file takes 3 arguments:")
	print("1. Image to get feature vector from")
	print("2. The type of features to return (HOG or SIFT")
	print("3. Whether or not to display the feature vector(true or false)")
	sys.exit()

fileName = str(sys.argv[1])
typeOfFeatures = str(sys.argv[2])
display = False
if(str(sys.argv[3]) == 'true'):
	display = True

image = utils.getGreyScaleImage(fileName)
if typeOfFeatures == 'HOG':
	image = utils.downScaleImage(image)
	fd = utils.getHOGFD(image)
else:
	sift = cv2.xfeatures2d.SIFT_create()
	fd = utils.getSIFTFD(image, sift)

if display:
	print(fd)
	print(len(fd))
	print(type(fd))