import cv2

# Download XML file for trained frontal face data from opencv:
# https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose image to detect faces in
img = cv2.imread('98Bulls.webp')

# Convert Image to Grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)

# Draw rectangle around faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Face Detector', img)
cv2.waitKey()

print("Code Completed")
