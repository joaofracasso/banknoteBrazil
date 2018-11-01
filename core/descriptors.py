import numpy as np
import cv2

# Return the descriptors found in the image
def sift(img):
    # Convert image to black and white
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(gray, None)

    # Return the descriptors
    return (keyPoints, descriptors)