import numpy as np
import scipy.spatial
import cv2
import descriptors

tree = []

img = cv2.imread('../data/synthetic/2reais.jpg')
kp, desc = descriptors.sift(img)
tree.append(scipy.spatial.KDTree(desc))

img = cv2.imread('../data/synthetic/5reais.jpg')
kp, desc = descriptors.sift(img)
tree.append(scipy.spatial.KDTree(desc))

img = cv2.imread('../data/synthetic/10reais.jpg')
kp, desc = descriptors.sift(img)
tree.append(scipy.spatial.KDTree(desc))

img = cv2.imread('../data/synthetic/20reais.jpg')
kp, desc = descriptors.sift(img)
tree.append(scipy.spatial.KDTree(desc))

img = cv2.imread('../data/synthetic/50reais.jpg')
kp, desc = descriptors.sift(img)
tree.append(scipy.spatial.KDTree(desc))

img = cv2.imread('../data/synthetic/100reais.jpg')
kp, desc = descriptors.sift(img)
tree.append(scipy.spatial.KDTree(desc))


def find(img):
    _, desc = descriptors.sift(img)


    img = cv2.drawKeypoints(img, kp, None, (100, 100, 100))
    cv2.imwrite('./test.png', img)

    for t in tree:
            
        # Search on tree using euclidian distance
        d, i = t.query(desc, 1)


    # Create tuples with values
    #array = np.array((i, np.arange(len(d)), d)).T

    # Clear all the matches over the treshold
    #array = array[array[:,2] < treshold]