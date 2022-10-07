# Pedestrian and Car Detection using OpenCV in Python

import cv2
import numpy as np

# Capture frames from a video
# Enter a mp4 file to run this program on
cap = cv2.VideoCapture("")

# Trained XML classifiers describes some features of some object we want to detect
# Download and locate cars_1.xml on your machine
car_cascade = cv2.CascadeClassifier("cars_1.xml")

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # To draw a rectangle in each cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frames, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frames, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)

    # frames = cv2.resize(frames, (600, 600))
    # Display frames in a window
    cv2.imshow('video2', frames)

    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()
