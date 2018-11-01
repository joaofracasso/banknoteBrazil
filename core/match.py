import numpy as np
import scipy.spatial
import cv2

# Faster implementation of matching process
def match_tree(desc1, desc2, treshold):
    # Create search tree
    kdtree = scipy.spatial.KDTree(desc1)

    # Search on tree using euclidian distance
    d, i = kdtree.query(desc2, 1, distance_upper_bound=treshold)

    # Create tuples with values
    array = np.array((i, np.arange(len(d)), d)).T

    # Clear all the matches over the treshold
    array = array[array[:,2] < treshold]

    # Create OpenCV Structure DMatch (used to generate images)
    matches = []
    for match in array:
        x = []
        x.append(cv2.DMatch(int(match[0]), int(match[1]), match[2]))
        matches.append(x)

    return matches