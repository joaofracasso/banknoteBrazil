import descriptors
import cv2
import argparse
import numpy as np
import notes

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	            help="Path to Input Image")

args = ap.parse_args()

img = cv2.imread(args.image)
notes.find(img)