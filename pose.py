import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Pose module
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set up the Pose detector
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the camera
cap = cv2.VideoCapture(0)
time.sleep(2)  # Camera warm-up

current_keys = set()

def update_keys(action):
    global current_keys

    keys_to_press = set()
    if action == "left":
        keys_to_press.add("left")
    elif action == "right":
        keys_to_press.add("right")
    elif action == "jump":
        keys_to_press.add("up")

    for key in keys_to_press - current_keys:
        pyautogui.keyDown(key)

    for key in current_keys - keys_to_press:
        pyautogui.keyUp(key)

    current_keys = keys_to_press

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
        left_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y

        action = None

        # Jump: both wrists above both shoulder
        if left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y:
            action = "jump"
        # Move Left: left wrist above left shoulder
        elif left_wrist_y < left_shoulder_y:
            action = "left"
        # Move Right: right wrist above right shoulder
        elif right_wrist_y < right_shoulder_y:
            action = "right"
        else:
            action = "idle"

        update_keys(action)

        # Optional: show action on frame
        if action:
            cv2.putText(image, f"Action: {action}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the image
    cv2.imshow("Pose Key Sender", image)

    # Press 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()