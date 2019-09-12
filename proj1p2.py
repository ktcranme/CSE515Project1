import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

#example commands:
#python proj1p2.py ../Images/HandsMed/ ../CSV/HOGMed.csv HOG
#python proj1p2.py ../Images/HandsMed/ ../CSV/SIFTMed.csv SIFT

if len(sys.argv) != 4:
	print("Error: This file takes 3 arguments:")
	print("1. The directory where the images are stored")
	print("2. The filename (including desired path) of location of CSV that will contain feature vectors (must include .csv)")
	print("3. The type of features to compare (HOG or SIFT)")
	sys.exit()

filePath = str(sys.argv[1])
outfile = str(sys.argv[2])
typeOfFeature = str(sys.argv[3])

#get all files in the folder
files = utils.getListOfImageFilenames(filePath)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
data = []
#loop through all hands, storing them in data to be put into csv later
for file in files:
	#this file was present for some reason
	if(file == '.DS_Store'):
		continue
	image = utils.getGreyScaleImage(filePath + file)
	if(typeOfFeature == 'HOG'):
		downscaledImage = utils.downScaleImage(image)
		fd = utils.getHOGFD(downscaledImage)
	else:
		fd = utils.getSIFTFD(image, sift)
	entry = []
	entry.append(file)
	entry.extend(fd)
	data.append(entry)

utils.writeToCSV(data, outfile)




#implement matching by myself