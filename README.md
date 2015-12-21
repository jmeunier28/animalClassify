# animalClassify
classify animals in a video 

In order to use classify.py you must create a image data set of different photographs of animals.  then use the conversion.py to create a mask for each image to be used as an input to the RBGHistogram class from pyimagesearch.  This will have difficulty classifying animals of similar colors so another classifier can be used in conjuction with this to improve the results. 

Run the classify.py script with the command: python classify.py -v videoname

Run the conversion.py script with the command: python conversion.py -i imagepath -m maskpath
