# Python program that uses openCV to create a face, eye and hand detection model, 
# plotting all relevant landmark points and connections to a live video stream. 
# Further differentiating between palm, thumb and fingertips.

# Import necessary modules
import cv2
import mediapipe as mp
import time

# Capture video from webcam but other video files are possible
cap = cv2.VideoCapture(0)

# Face and eye training cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


# Define MP hands
mpHands = mp.solutions.hands

# Define parameters for MP hands object - Keep default values
hands = mpHands.Hands()

# Define MP method to draw hands
mpDraw = mp.solutions.drawing_utils

# Frame Rate Variables
pTime = 0
cTime = 0

while True:
    # Read in frame
    success, img = cap.read()

    # Convert img to RGB colorscale
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert frame to gray frame
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Process results
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    # Determine face and eye coordinates
    face_coordinates = face_cascade.detectMultiScale(gray_frame)
    eye_coordinates = eye_cascade.detectMultiScale(gray_frame)

    # Graphs squares around face and eyes
    for (fx, fy, fw, fh) in face_coordinates:
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    for (ex, ey, ew, eh) in eye_coordinates:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    # Check to see if there is multiple hands
    # Then draw the points and corresponding connections
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Print Landmark ID for hand tracking data
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                # Mathematically Determine LandMark Center
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx,cy)
                # Determine if first landmark (palm) is being displayed, then differentiate it
                if id == 0:
                    cv2.circle(img, (cx,cy), 20, (255,0,255), cv2.FILLED)
                # Detemine if thumb (id=4) is being displayed
                if id == 4 :
                    cv2.circle(img, (cx, cy), 10, (255, 127, 0), cv2.FILLED)
                # Detemine if fingertips are being displayed
                if id == 8 or id == 12 or id == 16 or id == 20 :
                    cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Current Time
    cTime = time.time()

    # Calculate Frames Per Second (FPS)
    fps = 1/(cTime-pTime)

    # Previous time now becomes current time
    pTime = cTime

    # Display FPS
    cv2.putText(img, "FPS ", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cv2.putText(img,str(int(fps)), (10,80), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    # Display frame
    cv2.imshow("Hand and Face Detector", img)

    # Update frame every millisecond
    key=cv2.waitKey(1)

    # Hit q / Q key to quit out of the program
    if key==81 or key==113:
        break

print("Code Completed")
