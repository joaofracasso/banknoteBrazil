def drawKeypoints(frame, kps, BGR, size):
    for i in range(len(kps)):
        for j in range(len(kps[i])):
            x = math.floor(kps[i, j, 0])
            y = math.floor(kps[i, j, 1])

            frame = cv2.circle(frame, (x, y), size, BGR, -1)

    return frame