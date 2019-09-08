import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

if len(sys.argv) != 7:
	print("Error: This file takes 6 arguments:")
	print("1. Image to be compared (filename only, filepath will be appended from arg 4)")
	print("2. Number of most similar images to display")
	print("3. The type of features to compare (HOG or SIFT")
	print("4. The filepath + filename of the CSV file where the image feature data is stored")
	print("5. The directoy where the images are stored")
	print("6. Either 'true' to display similar images, or 'false' to only print similar image names")
	sys.exit()
#python proj1p3.py Hand_0000026.jpg 5 HOG ../CSV/HOG.csv ../Images/HandsSmall/ true

#handle command line args
fileName = str(sys.argv[1])
try:
	n = int(sys.argv[2])
except ValueError as e:
	print("The second argument must be a valid integer")
	sys.exit()
featureType = str(sys.argv[3])
CSVfile = str(sys.argv[4])
imagePath = str(sys.argv[5])
if(str(sys.argv[6]) == "true"):
	display = True
else:
	display = False

#extract data from CSV
df = pd.read_csv(CSVfile)

#get feature vector for given image
fvdf = df[df[df.columns[0]] == fileName]
fv = fvdf.values.flatten()
fv = fv[1:]
fv = fv.astype(np.float)

#get rest of data, keep filenames in there for now, comparing functions handles them
data = df.values



similarImages = utils.findNClosestHOG(fv, data, n)
print(similarImages)
if display:
	#displays the image to be compared with
	plt.imshow(utils.getRGBImage(imagePath + fileName))
	plt.show()
	
	#display the n most similar images in a row
	for image in similarImages:
		plt.imshow(utils.getRGBImage(imagePath + image))
		plt.show()
