import numpy as np
import cv2
from os.path import isfile, join
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from skimage.measure import block_reduce
import os
import csv


def getListOfImageFilenames(filePath):
	return [f for f in os.listdir(filePath) if isfile(join(filePath, f))]

#returns 3D matrix
#Dimensions: (x), (y), (tuple of RGB)
def getRGBImage(imageName):
	image = cv2.imread(imageName)
	return image

#returns 3D matrix
#Dimensions:(x), (y), (tuple of YUV)
def getYUVImage(imageName):
	#Y = 0.299R + 0.587G + 0.114B
	#U = 0.492(B-Y)
	#V = 0.877(R-Y)
	image = cv2.imread(imageName)
	YUVimage = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
	return image

def getGreyScaleImage(imageName):
	image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
	return image

#gets the feature vector for HOG
def getHOGFD(image):
	fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), block_norm='L2-Hys', cells_per_block=(2, 2), visualize=True, feature_vector=True)
	#plt.imshow(hog_image, cmap=plt.cm.gray) 
	#plt.show()
	return fd

def downScaleImage(image):
	return block_reduce(image, block_size=(10, 10), func=np.mean)


def writeToCSV(data, fileName):
	#with open(outfile, 'ab'):
	with open(fileName, 'w') as File:
	    writer = csv.writer(File)
	    writer.writerows(data)

def compareTwoHOG(fd1, fd2):
	l = fd1 - fd2
	l = np.square(l)
	l = np.sum(l)
	l = np.sqrt(l)
	return l

#returns an array containing the names for n most similar
#images contained in data
#input param data is immediate matrix from CSV, still has image name in first
#column and is of type list
def findNClosestHOG(fd, data, n):
	npData = np.array(data)
	npData = npData[:, 1:]
	npData = npData.astype(np.float)
	difference = npData - fd
	squared = np.square(difference)
	summed = np.sum(squared, axis=1)
	print("looking for ", n, " closest images...");

	#sorts the array, takes a bit to do
	ind = np.argpartition(summed, n+1)[:n+1]
	ind = ind[np.argsort(summed[ind])]

	#if chosen image is in data still
	#prune it from list here
	if(summed[ind[0]] == 0):
		ind = ind[1:]
	else:
		ind = ind[:-1]

	#get list of image names instead of indices
	result = []
	for i in ind:
		result.append(data[i][0])
		print(summed[i])

	return result




