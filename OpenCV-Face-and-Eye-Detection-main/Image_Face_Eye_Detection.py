import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Choose image to detect faces in
img = cv2.imread('michael-jordan.jpg')

# Convert Image to Grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
face_coordinates = face_cascade.detectMultiScale(grayscaled_img)
eye_coordinates = eye_cascade.detectMultiScale(grayscaled_img)

# Draw rectangle around faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

for (x, y, w, h) in eye_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 127, 255), 2)

cv2.imshow('Face and Eye Detector', img)
cv2.waitKey()

print("Code Completed")
