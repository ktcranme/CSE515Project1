import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import csv
import cv2

if len(sys.argv) != 9:
	print("Error: This file takes 8 arguments:")
	print("1. Image to be compared filename")
	print("2. Image to be compared filepath")
	print("3. Number of most similar images to display")
	print("4. The type of features to compare (HOG or SIFT")
	print("5. The filepath + filename of the CSV file where the image feature data is stored")
	print("6. The directoy where the images are stored")
	print("7. Either 'true' to display similar images, or 'false' to only print similar image names")
	print("8. Either 'true' to display a graph of the image differences or 'false' to not display graph")
	sys.exit()
#python proj1p3.py Hand_0000026.jpg ../Images/HandsSmall/ 5 HOG ../CSV/HOG.csv ../Images/HandsSmall/ true true
#python proj1p3.py Hand_0000003.jpg ../Images/HandsSmall/ 5 SIFT ../CSV/SIFTSmall.csv ../Images/HandsSmall/ true true

#handle command line args
fileName = str(sys.argv[1])
filePath = str(sys.argv[2])
try:
	n = int(sys.argv[3])
except ValueError as e:
	print("The second argument must be a valid integer")
	sys.exit()
featureType = str(sys.argv[4])
CSVfile = str(sys.argv[5])
imagePath = str(sys.argv[6])
if(str(sys.argv[7]) == "true"):
	display = True
else:
	display = False
if(str(sys.argv[8]) == "true"):
	displayGraph = True
else:
	display = False

if featureType == 'HOG':
	#extract data from CSV
	df = pd.read_csv(CSVfile)

	#get feature vector for given image
	image = utils.getGreyScaleImage(filePath + fileName)
	image = utils.downScaleImage(image)
	fv = utils.getHOGFD(image)

	#get rest of data, keep filenames in there for now, comparing functions handles them
	data = df.values


	similarImages, distances = utils.findNClosestHOG(fv, data, n)
	print(similarImages)
else:
	#extract data from csv
	#can't use pandas cuz rows are variable length (sad face)
	data = []
	names = []
	with open(CSVfile,'r') as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			if(row[0] == fileName):
				image = utils.getGreyScaleImage(filePath + fileName)
				sift = cv2.xfeatures2d.SIFT_create()
				fv = utils.getSIFTFD(image, sift)
			else:
				arr = np.array(row[1:], dtype=np.float32)
				k = int(len(arr) / 132)
				arr = arr.reshape((k, 132))
				data.append(arr)
				names.append(row[0])

	similarImages, distances = utils.findNClosestSIFT(fv, data, names, n)
	print(similarImages)



if display:
	#displays the image to be compared with
	plt.imshow(utils.getRGBImage(imagePath + fileName))
	plt.show()

	#display the n most similar images in a row
	for image in similarImages:
		plt.imshow(utils.getRGBImage(imagePath + image[0]))
		plt.show()

if displayGraph:
	# the histogram of the data
	plt.hist(distances, 30, density=True, facecolor='g', alpha=0.75)
	plt.xlabel('Distance')
	plt.ylabel('Count')
	plt.title('Distances from ' + fileName)
	plt.xlim(0, 1)
	plt.show()

