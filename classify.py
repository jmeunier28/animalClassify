# USAGE: python classify.py --images dataset/images --masks dataset/masks -v videopath.mp4
# Author: Jmeunier
# use random forest classifier to classify differnet animals from image dataset
# use this as input to camshift algorithm in video file
from __future__ import print_function
# import the necessary packages
from pyimagesearch.rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import imutils
from time import time
import argparse
import glob
import cv2
import json

###################################################################################
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required = True,
#	help = "path to the image dataset")
#ap.add_argument("-m", "--masks", required = True,
#	help = "path to the image masks")
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

#grab the image and mask paths
#imagePaths = sorted(glob.glob(args["images"] + "/*.jpg"))
#masksPaths = sorted(glob.glob(args["masks"] + "/*.jpg"))

# be real salty if theres no video:
if args.get("video", None) is None:
	print('i would like a video plz')
else:
	camera = cv2.VideoCapture(args["video"])

#hard coded image paths
imagePaths= sorted(glob.glob('datasets2/images'+"/*.jpg"))
masksPaths = sorted(glob.glob('datasets2/masks'+"/*.jpg"))

###################################################################################
# Train classifier based off of input data
# initialize the list of data and class label targets
data = []
target = []

# make new feature descripter object
# each color channel uses 2 bins
desc = RGBHistogram([2,2,2])

#start timing amount of time to train data and classify
t0 = time()

# loop over the image and masks paths
for (imagePath, maskPath) in zip(imagePaths, masksPaths):
	# load the images in the path
	image = cv2.imread(imagePath)
	mask = cv2.imread(maskPath)
	# change mask to gray scale
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	#check to make sure that the image and mask are of equal size
	# or else an error will occur when extracting features using
	#calcHist openCV function
	if(mask.shape != (500,255)):
		mask = cv2.resize(mask,(500,255))
	if(image.shape != (500,255,3)):
		image = cv2.resize(image,(500,255))
	# extract features and append to data list
	features = desc.describe(image,mask)
	#features = features.reshape(1,-1)
	data.append(features)
	# get target name aka the name of the animal in the picture
	# by looking at the name of the image
	target.append(imagePath.split("_")[-2])

# grab the unique target names and encode the labels
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

# construct the training and testing splits
# test_size = proportion of data to test on
# random_state = Pseudo-random number generator state used for random sampling
(trainData, testData, trainTarget, testTarget) = train_test_split(data, target,
	test_size = 0.30, random_state = 42)


# train the classifier using random forest
# ensemble classifier for multi-class classification
# n_esitmators = number of trees in forest
model = RandomForestClassifier(n_estimators = 25, random_state = 84)
model.fit(trainData, trainTarget)

# print classification accuracy report
print(classification_report(testTarget, model.predict(testData),
	target_names = targetNames))

# print amount of time this took
print("done in %.3fs"%(time()-t0))

#show how the classifier works by choosing some random images
#and predicting what that animal is five different times
'''
for i in np.random.choice(np.arange(0, len(imagePaths)), 5):

	# grab the image and mask paths
	imagePath = imagePaths[i]
	maskPath = masksPaths[i]

	# load the image and mask
	image = cv2.imread(imagePath)
	mask = cv2.imread(maskPath)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	#check to make sure dat sizing is right
	if(mask.shape != (500,255)):
		mask = cv2.resize(mask,(500,255))
	if(image.shape != (500,255,3)):
		image = cv2.resize(image,(500,255))

	# extract image descriptor
	features = desc.describe(image, mask)
	features = features.reshape(1,-1)
	# tell me what kind of animal it is plz
	animal = le.inverse_transform(model.predict(features))[0]
	print(imagePath)
	print ("I think this animal is a %s" % (animal.upper()))

	cv2.imshow("image", image)
	cv2.waitKey(0)

###################################################################################
# Classify the animals in the video
'''
frame_counter = 0 # count the number of frames we are working wtih
outputFrameIndices=[]
animalCount = 0
t1 = time() # initialize the time
animalArr = []
outputArr = [[]]
while(True):
	ret,frame = camera.read() # begin reading video frames
	#if there are no frames left to get then the video is over
	if frame is None:
		print('thats the end!')
	    	break
	# create a mask for the frame and use it to extract features and
	# predict what kind of animal is in the frame
	_ ,frameMask = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
	frameMask = cv2.cvtColor(frameMask, cv2.COLOR_BGR2GRAY)
	features=desc.describe(frame,frameMask)
	# must reshape features bc Passing 1d arrays as data is deprecated
	# .17 and will cause ValueError in .19
	features = features.reshape(1,-1)
	animal = le.inverse_transform(model.predict(features))[0]
	animalArr.append(str(animal.upper()))
	print(str(animal.upper()))

	frame_counter=frame_counter+1 # increase frame count
	outputFrameIndices.append(frame_counter) # put into 1 d array

	if cv2.waitKey(1) & 0xFF == ord('q'): # if exit key is pressed end video
		break
	cv2.imshow("hello animlas",frame) #display the video

#take the arrays and make them vectors
animalArr = np.array([[animalArr]])
outputFrameIndices = np.array([[outputFrameIndices]])
#put the animal name and corresponding frame count and make 2d array
#print(animalArr.shape)
#print(outputFrameIndices.shape)
outputArr=np.hstack((animalArr,outputFrameIndices))
#print(outputArr)

#print ('total number of frames: ' + str(len(outputFrameIndices)))
#print ('total num of animals: '+str(len(animalArr)))
camera.release() #release the camera
###################################################################################
