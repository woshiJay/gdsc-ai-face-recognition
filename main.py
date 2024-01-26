import cv2
import cvzone
import os
import pickle
import face_recognition
import numpy as np

capture = cv2.VideoCapture(0)

capture.set(3,1280)
capture.set(4,720)

backgroundImg = cv2.imread("Resources/Background.png")

folderPathMode = 'Resources/Modes'
listPathMode = os.listdir(folderPathMode)
imgListMode= []

for path in listPathMode:
    imgListMode.append(cv2.imread(os.path.join(folderPathMode,path)))

encodingsFile = open("EncodingsFile.p", "rb")
encodingsListWithIDs = pickle.load(encodingsFile)
encodingsFile.close()

encodingsListKnown, studentIDs = encodingsListWithIDs
print(studentIDs)

while True:
    success,image=capture.read()
    image = cv2.resize(image, (640,480))
    image = cv2.flip(image, 1)
    backgroundImg[162:162+480, 55:55+640] = image
    backgroundImg[44:44 + 633, 808:808 + 414] = imgListMode[0]

    # making frame 1/4 of the original size
    smallImage = cv2.resize(image, (0,0), None, 0.25, 0.25)
    # converting image to rgb color format
    smallImage = cv2.cvtColor(smallImage, cv2.COLOR_BGR2RGB)

    # finding the face in webcam frame
    faceCurrentFrame = face_recognition.face_locations(smallImage)
    # encode the face found
    encodeCurrentFrame = face_recognition.face_encodings(smallImage, faceCurrentFrame)

    # showing a window for webcam
    # cv2.imshow("Attendance System", image)


    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodingsListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodingsListKnown, encodeFace)
        print("Matches", matches)
        print("Face Distance", faceDistance)

        matchIndex = np.argmin(faceDistance)
        print("Match Index", matchIndex)

        if matches[matchIndex]:
            print("Registered Student Detected")
            print("Student ID", studentIDs[matchIndex])

        # creating four points to map the face
        y1, x1, x2, y2 = faceLocation
        # resizing it to actual feed
        y1, x1, x2, y2 = y1*4, x1*4, x2*4, y2*4
        # creating a box around the face
        bbox = 55+x1, 162 + y1, x2 - x1, y2 - y1
        # having the rectangle follow around our face
        # rt = 0 means the box is not outlined
        backgroundImg = cvzone.cornerRect(backgroundImg, bbox, 20, rt=0)

    # showing a window for backgroundImg
    cv2.imshow("Attendance System", backgroundImg)
    cv2.waitKey(1)