#USAGE: python -i datasets2/images
# Author: Jmeunier
# take image data and create masks for each image
# do this by creating a binary threshold and then writing
# the new image to a directory

import cv2
import numpy as np
import glob
import os

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True,
	help = "path to the image dataset")
#ap.add_argument("-m", "--masks", required = True,
#	help = "path to the image masks")
args = vars(ap.parse_args())

# grab the image paths
imagePaths = sorted(glob.glob(args["images"] + "/*.jpg"))
imgnum = len(imagePaths)
dirname = 'datasets2'
dirname2 = 'masks'
for i in range(imgnum):
	im = cv2.imread(imagePaths[i],0)
	#generate binary masks
	ret,thresh = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
	target = imagePaths[i].split("_")[-2]
	# write new images to correct directory
	newImg = cv2.imwrite(os.path.join(dirname,target+"_"+str(i)+".jpg"),thresh)
