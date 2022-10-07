# Python pose estimation models using the openCV and mediapipe libraries.

# Import necessary libraries
import cv2
import mediapipe as mp
import time

# Define MP Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Define mpDraw
mpDraw = mp.solutions.drawing_utils

# Capture video from webcam OR video files are possible
cap = cv2.VideoCapture('ConorMcGregorFinishes.mp4')

pTime=0

while True:
    # Read in frame
    success, img = cap.read()

    # Convert img to RGB colorscale
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process results
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)

    # Check if pose is detected, then draw pose if detected, and print landmark values
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
    # Update frame every millisecond
    key=cv2.waitKey(1)

    cTime = time.time() # Current Time
    fps = 1/(cTime-pTime) # Calculate Frames Per Second (FPS)
    pTime = cTime # Previous time now becomes current time

    # Display FPS
    cv2.putText(img, "FPS ", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cv2.putText(img, str(int(fps)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    # Make it so video is displayed in full screen
    #cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Display frame
    cv2.imshow("Image", img)

    # Hit q / Q key to quit out of the program
    if key==81 or key==113:
        break

print("Code Completed")
