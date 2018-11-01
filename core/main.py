import descriptors
import cv2
import utils
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	            help="Path to Input Image")

args = ap.parse_args()

img = cv2.imread(args.image)
kp, desc = descriptors.sift(img)
img = utils.drawKeypoints(img, kp, (100, 100, 100), 10)
cv2.imwrite('./test.png')