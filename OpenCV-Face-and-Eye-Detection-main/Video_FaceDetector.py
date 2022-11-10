import cv2

# Download XML file for trained frontal face data from GitHub:
# https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture Video from Webcam
# 0 for webcam BUT can specify other video files
webcam = cv2.VideoCapture(0)

while True:
    # Read current frame
    succesful_frame_read, frame = webcam.read()

    # Detect Face Cooridnates
    face_coordinates = trained_face_data.detectMultiScale(frame)

    # Draw rectangle around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Webcam Face Detector", frame)
    key=cv2.waitKey(1)

    # Hit q key to quit out of the program
    if key==81 or key==113:
        break

print("Code Completed")
