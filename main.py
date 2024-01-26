import cv2
import os

capture = cv2.VideoCapture(0)

capture.set(3,1280)
capture.set(4,720)

backgroundImg = cv2.imread("Resources/Background.png")

folderPathMode = 'Resources/Modes'
listPathMode = os.listdir(folderPathMode)
imgListMode= []

for path in listPathMode:
    imgListMode.append(cv2.imread(os.path.join(folderPathMode,path)))

while True:
    success,image=capture.read()

    image = cv2.resize(image, (640,480))
    backgroundImg[162:162+480, 55:55+640] = image

    # showing a window for webcam
    # cv2.imshow("Attendance System", image)

    # showing a window for backgroundImg
    cv2.imshow("Attendance System", backgroundImg)
    cv2.waitKey(1)