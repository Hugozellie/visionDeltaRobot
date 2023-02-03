import cv2 as cv
import numpy as np
import imutils as im

output = []

key = cv. waitKey(1)
webcam = cv.VideoCapture(1, cv.CAP_DSHOW)

webcam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

r, frame = webcam.read()

while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)
        cv.imshow("Capturing", frame)
        key = cv.waitKey(1)
        
        if key == ord('s'):
            frameCrop = frame[150:550, 400:860]
            cv.imwrite(filename='saved_img.png', img=frameCrop)
            webcam.release()
            print("Image saved!")
            break
    
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv.destroyAllWindows()
        break

colourMaskColors = {
    "rood"   : [(9, 255, 255), (0, 175, 150)], #
    "groen"  : [(95, 255, 255),(80, 60, 60)], #
    "wit"    : [(255, 30, 255), (0, 0, 100)], #
    "blauw"  : [(110, 255, 255),(96, 100, 100)], #
    "paars"  : [(140, 255, 255),(115, 50, 100)], 
    "oranje" : [(18, 174, 255), (0, 50, 100)], #
    "geel"   : [(30, 255, 255), (19, 100, 100)] #
}

img = cv.imread("saved_img.png", 1)

imgBlur1 = cv.GaussianBlur(img, (9,9), 0)
imgBlur2 = cv.medianBlur(imgBlur1, 25)
imgHSV   = cv.cvtColor(imgBlur2, cv.COLOR_BGR2HSV)    

for i in colourMaskColors.keys():
    mask = cv.inRange(imgHSV, colourMaskColors[i][1], colourMaskColors[i][0])
    imgGoed = cv.resize(mask, None, fx=1, fy=1, interpolation=cv.INTER_LINEAR)

    contours = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = im.grab_contours(contours)

    for cont in contours:
        m = cv.moments(cont)
        area = m["m00"]
        
        if area > 1500:
            cx = int(m["m10"] / area)
            cy = int(m["m01"] / area)

            rect = cv.minAreaRect(cont)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            r = round(rect[2])
            output.append([i, cx,cy, r])

            cv.drawContours(img, [cont], -1, (255,255,255), 3, cv.LINE_AA)
            cv.drawContours(img, [box], -1, (0,255,0), 3, cv.LINE_AA)
            cv.circle(img, (cx,cy), 2, (255,255,255), -1)

    cv.imshow('result',imgGoed)
    cv.waitKey()

print(output)

imgGoed = cv.resize(img, None, fx=1, fy=1, interpolation=cv.INTER_LINEAR)

cv.imshow('plaatje',imgGoed)
cv.waitKey(0)
cv.destroyAllWindows()