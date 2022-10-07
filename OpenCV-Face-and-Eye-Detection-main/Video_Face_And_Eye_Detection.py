import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    succesful_frame_read, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = face_cascade.detectMultiScale(gray_frame)
    eye_coordinates = eye_cascade.detectMultiScale(gray_frame)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in eye_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 127, 255), 2)

    cv2.imshow("Webcam Face Detector", frame)
    key=cv2.waitKey(1)

    # Hit q key to quit out of the program
    if key==81 or key==113:
        break

print("Code Completed")
