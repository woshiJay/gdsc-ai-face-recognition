import cv2

capture = cv2.VideoCapture(0)

capture.set(3, 1280)
capture.set(4, 720)

while True: 
    success, image = capture.read()
    cv2.imshow("Attendance System", image)
    cv2.waitKey(1)