# OpenCV Mediapipe Face Mesh Detection 
# Version 1

# Import necessary libraries
import cv2
import mediapipe as mp

# Define face mesh solutions model
mp_face_mesh = mp.solutions.face_mesh

# Define drawing tools to later draw facemesh
mp_drawing = mp.solutions.drawing_utils

# Specs for our drawing tools
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255,0,0))

# Capture webcam input
cap = cv2.VideoCapture(0)

# Define minium detection and tracking confidence
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
        # Apply below code while webcam is on - webcam input manipulation
        while cap.isOpened():
            # Read in webcam image
            success, image = cap.read()
            
            # Process image through facemesh
            results = face_mesh.process(image)

            # Allows for multiple faces to be detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Print landmark coordinates in console
                    print(face_landmarks)

                    # Draw face mesh onto webcam input image
                    mp_drawing.draw_landmarks(image, landmark_list=face_landmarks,
                                              connections=mp_face_mesh.FACEMESH_CONTOURS,
                                              landmark_drawing_spec=drawing_spec,
                                              connection_drawing_spec=drawing_spec)

            # Display frame
            cv2.imshow("FaceMesh App", image)

            # Update frame every millisecond
            key = cv2.waitKey(1)

            # Hit q / Q key to quit out of the program
            if key == 81 or key == 113:
                break

# Print to indicate code is done running
print("Code Completed")
