import cv2
import mediapipe as mp
import keyboard

# Initialize MediaPipe Pose module
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set up the Pose detector
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("No camera input detected.")
        break

    # Flip the image horizontally for a mirror view
    image = cv2.flip(image, 1)

    # Convert the image to RGB for MediaPipe processing
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get landmark positions
        landmarks = results.pose_landmarks.landmark

        # Get the Y-coordinates of wrists and nose
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y

        # Check if both wrists are higher (smaller Y) than the nose
        if left_wrist_y < nose_y and right_wrist_y < nose_y:
            cv2.putText(image, "Jump!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            keyboard.press_and_release('space')  # Trigger space key

    # Display the image
    cv2.imshow('Pose Control', image)

    # Press 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
