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
	image = cv2.imread(imageName, cv2.IMREAD_COLOR)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
	image = cv2.imread(imageName, 0)
	return image

#gets the feature vector for HOG
def getHOGFD(image):
	fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), block_norm='L2-Hys', cells_per_block=(2, 2), visualize=True, feature_vector=True)
	#plt.imshow(hog_image, cmap=plt.cm.gray) 
	#plt.show()
	return fd

#returns a flattened feature vector of size K x 132 where
#K is the number of keypoints found on the image
#the first 4 elements of the fv are x, y, angle, scale
#the next 128 elements are the histogram or oriented gradients
def getSIFTFD(image, sift):
	kp, des = sift.detectAndCompute(image,None)
	keypoints = []
	for i in range(len(kp)):
		keypoints.extend([kp[i].pt[0], kp[i].pt[1], kp[i].angle, kp[i].size])
		keypoints.extend(des[i])

	return np.array(keypoints)


def downScaleImage(image):
	return block_reduce(image, block_size=(10, 10), func=np.mean)


def writeToCSV(data, fileName):
	#with open(outfile, 'ab'):
	with open(fileName, 'w') as File:
	    writer = csv.writer(File)
	    writer.writerows(data)

def EuclideanDistance(v1, v2):
	l = v1 - v2
	l = np.square(l)
	l = np.sum(l)
	l = np.sqrt(l)
	return l

def compareTwoHOG(fd1, fd2):
	return EuclideanDistance(fd1, fd2)

def findDistanceBetweenTwoKeyPoints(kp1, kp2):
	return EuclideanDistance(kp1, kp2)

def getClosestMatches(kp, fd2, n):
	mtx = fd2 - kp
	mtx = np.square(mtx)
	mtx = np.sum(mtx, axis=1)
	mtx = np.sqrt(mtx)
	matches = []
	maximum = np.amax(mtx)
	for i in range(n):
		minimum = np.amin(mtx)
		minIndex = np.where(mtx == np.amin(mtx))[0][0]
		mtx[minIndex] = maximum		#this is so we can find the next lowest min
		matches.append([minIndex, minimum])
	return np.array(matches)

#returns an array where each element is a set of n matches, 
#each match having the list: [queryIdx, trainIdx, distance]
def getKeypointMatches(fd1, fd2, n):
	matches = []
	for i in range(len(fd1)):
		t = getClosestMatches(fd1[i], fd2, n) #t is a n x 2 matrix with trainIdx and distance as columns
		iArr = np.array([i] * n)
		t = np.c_[iArr, t]	#append a column of i to t so that the query index is included in the match
		#print(t[0][0], t[0][1], t[0][2])
		matches.append(t)		#query index, train index, distance
	return matches

#matchCountWeight is for how much to weigh the amount
#of matches, 1 to only care about number of matches, 0 to only care about match
#distance average (not implemented yet)
#threshold is ...
def compareTwoSIFT(fd1, fd2, threshold=0.8, matchCountWeight=0.5):
	#get descriptors without x, y, angle, scale
	fd1 = np.array(fd1, dtype=np.float32)
	fd2 = np.array(fd2, dtype=np.float32)
	fd1 = fd1.reshape((int(len(fd1) / 132), 132))
	fd1 = np.delete(fd1, np.s_[0:4], 1)
	fd2 = np.delete(fd2, np.s_[0:4], 1)

	# BFMatcher with default params
	matches = getKeypointMatches(fd1, fd2, 2)

	# Apply ratio test
	good = []
	for m,n in matches:
		if m[2] < 0.8*n[2]:
			good.append([m[2]])

	good = np.array(good)
	distance = np.sqrt(np.square(good).sum()) #square them to punish extreme differences
	distanceAvg = (distance / len(good))


	return distanceAvg

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
	maxDistance = np.amax(summed)
	print(maxDistance)
	#normalize to 0 to 1 scale
	summed = summed / maxDistance
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
		result.append((data[i][0], summed[i]))

	return result, summed


def findNClosestSIFT(fd, data, fileNames, n):
	npData = np.array(data)

	distances = []
	for fd2 in data:
		distances.append(compareTwoSIFT(fd, fd2))

	distances = np.array(distances)
	maxDistance = np.amax(distances)
	#normalize to 0 to 1 scale
	distances = distances / maxDistance

	#sorts the array, takes a bit to do
	ind = np.argpartition(distances, n+1)[:n+1]
	ind = ind[np.argsort(distances[ind])]

	#if chosen image is in data still
	#prune it from list here
	if(distances[ind[0]] == 0):
		ind = ind[1:]
	else:
		ind = ind[:-1]

	#get list of image names instead of indices
	result = []
	for i in ind:
		result.append((fileNames[i], distances[i]))

	return result, distances




