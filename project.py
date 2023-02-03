import cv2 as cv
import numpy as np
import imutils as im

img = cv.imread("project/fotos/groen.jpg", 1)

imgBlur1 = cv.GaussianBlur(img, (9,9), 0)
imgGray = cv.cvtColor(imgBlur1, cv.COLOR_BGR2GRAY)
imgBlur2 = cv.medianBlur(imgGray, 25)

_,temp = cv.threshold(imgBlur2, 30, 255, cv.THRESH_BINARY)

imgGoed = cv.resize(temp, None, fx=0.3, fy=0.3, interpolation=cv.INTER_LINEAR)
cv.imshow('yeet',imgGoed)

contours = cv.findContours(temp, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
contours = im.grab_contours(contours)

for cont in contours:
    m = cv.moments(cont)
    area = m["m00"]
    
    if area > 800:
        cx = int(m["m10"] / area)
        cy = int(m["m01"] / area)

        cv.drawContours(img, [cont], -1, (255,255,255), 10, cv.LINE_AA)
        cv.circle(img, (cx,cy), 20, (255,255,255), -1)

imgGoed = cv.resize(img, None, fx=0.3, fy=0.3, interpolation=cv.INTER_LINEAR)

cv.imshow('plaatje',imgGoed)
cv.waitKey(0)
cv.destroyAllWindows()