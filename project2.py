import cv2 as cv
import numpy as np
import imutils as im

output = []

colourMaskColors = {
    "rood"   : [(7, 255, 255), (0, 150, 150)],
    "groen"  : [(71, 255, 255),(34, 60, 60)],
    "wit"    : [(255, 30, 255), (0, 4, 100)],
    "blauw"  : [(111, 255, 255),(91, 100, 100)],
    "paars"  : [(146, 255, 255),(126, 100, 100)],
    "oranje" : [(18, 255, 255), (8, 100, 100)],
    "geel"   : [(32, 255, 255), (19, 100, 100)]
}

img = cv.imread("project/fotos/DingGeheel2.PNG", 1)

imgBlur1 = cv.GaussianBlur(img, (9,9), 0)
imgBlur2 = cv.medianBlur(imgBlur1, 25)
imgHSV   = cv.cvtColor(imgBlur2, cv.COLOR_BGR2HSV)    

for i in colourMaskColors.keys():
    mask = cv.inRange(imgHSV, colourMaskColors[i][1], colourMaskColors[i][0])
    imgGoed = cv.resize(mask, None, fx=0.2, fy=0.2, interpolation=cv.INTER_LINEAR)

    contours = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = im.grab_contours(contours)

    for cont in contours:
        m = cv.moments(cont)
        area = m["m00"]
        
        if area > 20000:
            cx = int(m["m10"] / area)
            cy = int(m["m01"] / area)

            rect = cv.minAreaRect(cont)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            r = round(rect[2])
            output.append([i, cx,cy, r])

            cv.drawContours(img, [cont], -1, (255,255,255), 10, cv.LINE_AA)
            cv.drawContours(img, [box], -1, (0,255,0), 5, cv.LINE_AA)
            cv.circle(img, (cx,cy), 20, (255,255,255), -1)

    cv.imshow('yeet',imgGoed)
    cv.waitKey()

print(output)

imgGoed = cv.resize(img, None, fx=0.2, fy=0.2, interpolation=cv.INTER_LINEAR)

cv.imshow('plaatje',imgGoed)
cv.waitKey(0)
cv.destroyAllWindows()