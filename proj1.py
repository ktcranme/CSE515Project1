import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2


filePath = '../Images/Hands/'
files = utils.getListOfImageFilenames(filePath)
outfile = '../CSV/SIFTLarge.csv'

"""
plt.imshow(utils.downScaleImage(image))
plt.show()
"""

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
data = []
for file in files:
	if(file == '.DS_Store'):
		continue
	image = utils.getGreyScaleImage(filePath + file)
	#downscaledImage = utils.downScaleImage(image)
	entry = []
	entry.append(file)
	fd = utils.getSIFTFD(image, sift)
	entry.extend(fd)
	data.append(entry)

utils.writeToCSV(data, outfile)

