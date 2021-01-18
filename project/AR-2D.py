import cv2
import numpy as np

# Webcam
webcam = cv2.VideoCapture(0)
#imgwebcam = cv2.imread('input_img.jpg')
#imgaug = imgwebcam.copy()

# image target //  img to replace aka the 4 card
imgtarget = cv2.imread('target_img.jpg')
hT,wT,cT = imgtarget.shape

# image replace // the img u want to show on webcam aka the cube
imgreplace = cv2.imread('replace_img.jpg')
imgreplace = cv2.resize(imgreplace,(wT,hT))

# create ORB keypoint detector
orb = cv2.ORB_create(nfeatures=1000)

# calculate keypoint/feature and description (in the img target)
# detect/extract the keypoint feature with orb // then save to the description
kp1, des1 = orb.detectAndCompute(imgtarget,None) # kp: key point, des: description
# imgtarget = cv2.drawKeypoints(imgtarget,kp1,None)

while True:
    # detect/description keypoint/feature on the img web, to find the 4 card in the web img
    success , imgwebcam = webcam.read()
    imgaug = imgwebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgwebcam,None)

    # finding match between 2 img, the img target and the webcam
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2) # array with each value is pair of keypoint
    '''
    DMatch.distance - Distance between descriptors. The lower, the better it is.
    DMatch.trainIdx - Index of the descriptor in train descriptors
    DMatch.queryIdx - Index of the descriptor in query descriptors
    DMatch.imgIdx - Index of the train image.
    '''

    # Taking good keypoints/feature
    good=[]
    for m,n in matches:
        if m.distance < 0.75 * n.distance: # 0.75 is the best ratio, lower -> hard to detect . higher -> keypoints chaos
            good.append(m)
    #print(len(good))

    # compute Homography if enough matches are found (in this case, it's > 20)
    if len(good) > 20:
        # differenciate between source points and destination points
        srcpts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2) # array with 1 columm each value is (2,1) array
        dstpts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # compute Homography //estimate the flat surface need to replace
        matrix, mask = cv2.findHomography(srcpts, dstpts, cv2.RANSAC, 5)

        #create boundary // find the corner of the img, then use homo matrix just find, then create border around the target img just found/captured on the webcam
        pts = np.float32([[0,0],[0,hT-1],[wT-1,hT-1],[wT-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)
        img2 = cv2.polylines(imgwebcam,[np.int32(dst)], True, (255,0,255), 3)

        #create a replace img // paint the webcam background to black (0,0,0), then use homo border (img2)(the 4 card), replace the inside the border (the 4 card) to the replaced img (the cube) 
        imgwrap = cv2.warpPerspective(imgreplace, matrix, (imgwebcam.shape[1],imgwebcam.shape[0]))
        
        #making mask // paint all the mask to black (0)
        masknew = np.zeros((imgwebcam.shape[0],imgwebcam.shape[1]),np.uint8)
        #then find the homo border, use the position in the webcam, cover the border to the mask, then paint inside the that border in mask to white (1)
        cv2.fillPoly(masknew,[np.int32(dst)],(255,255,255))
        
        #inverse the color of the mask, black(0) to white(1) and the otherway around
        maskinv = cv2.bitwise_not(masknew)
        # and:  1(white back ground of the mask) * ~1(Webcam) = 1
        #       0(the black of the mask) * ~1(Webcam) = 0
        imgaug = cv2.bitwise_and(imgaug,imgaug, mask = maskinv)
        #or: 0(the black of the mask) * ~1(the replace img) = 1
        #    1(white back ground of the mask) * ~1(the replace img) = out of range -> 0
        imgaug = cv2.bitwise_or(imgwrap, imgaug)

    # show matches
    #imgfeartures = cv2.drawMatches(imgtarget, kp1, imgwebcam, kp2, good, None, flags=2)

    cv2.imshow('result',imgaug)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break