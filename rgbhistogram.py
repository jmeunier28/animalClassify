# import the necessary packages
# use calcHist to return feature descritor
# example taken from pyimagesearch 
import numpy as np
import cv2

class RGBHistogram:
	def __init__(self, bins):
		# store the number of bins the histogram will use
		self.bins = bins

	def describe(self, image, mask = None):

		hist = cv2.calcHist([image], [0, 1, 2],
			mask, self.bins, [0, 256, 0, 256, 0, 256])

		hist = cv2.normalize(hist,hist)

		# return out 3D histogram as a flattened array
		return hist.flatten()
