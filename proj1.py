import utils
import matplotlib.pyplot as plt
import numpy as np


filePath = '../Images/Hands/'
files = utils.getListOfImageFilenames(filePath)
outfile = '../CSV/HOGLarge.csv'

"""
plt.imshow(utils.downScaleImage(image))
plt.show()
"""

data = []
for file in files:
	if(file == '.DS_Store'):
		continue
	image = utils.getGreyScaleImage(filePath + file)
	downscaledImage = utils.downScaleImage(image)
	entry = []
	entry.append(file)
	fd = utils.getHOGFD(downscaledImage)
	entry.extend(fd)
	data.append(entry)

utils.writeToCSV(data, outfile)

