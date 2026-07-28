import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((3*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:3].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('/media/alex/Data/Alex_Beng/code/opencv_deploy/2022/opencv-4.x/samples/data/left*.jpg')
cp = cv.VideoCapture("/media/alex/Windows/Users/Alex Beng/Videos/Captures/genshin_cali.mp4")

while True:
    ret, frame = cp.read()
    img = frame

# for fname in images:
#     img = cv.imread(fname)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    dst = cv.cornerHarris(gray,2,9,0.03)
        #result is dilated for marking the corners, not important
    # dst = cv.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    cv.imshow('dst',img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
    continue


    corners = cv.goodFeaturesToTrack(gray,200, 0.0001, 50)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),3, (0,0,255),-1)
    
    cv.imshow("ya", img)
    cv.waitKey()
    continue

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (4,3), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (4,3), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
cv.destroyAllWindows()

