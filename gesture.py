import cv2
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib as plt
import time
import numpy as np
import yaml
from mediapipe.framework.formats import landmark_pb2


# Open the USB camera (usually 0 or 1 for the first camera, 2 or higher for additional cameras)
# If you have multiple cameras, try changing the index (0, 1, 2, etc.)
camera_index = 1
mvs = True
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) # Windows 可以用cv2.Cap_DSHOW顯示



if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get video properties for recording
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_file = f"gesture_recognition.mp4"
# Initialize video writer with H.264 codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))


with open("mvs.yaml", "r") as file:
            mvs_data = yaml.load(file, Loader=yaml.FullLoader)
            cameraMatrix = np.array(mvs_data['camera_matrix']['data'])
            cameraMatrix = cameraMatrix.reshape(3,3)
            distortion_coefficients = np.array(mvs_data['distortion_coefficients']['data'])
            distortion_coefficients = distortion_coefficients.reshape(1,5)

print("Press 'q' to quit.")
# Global variable to store the latest recognition result
latest_result = None

# Callback function that updates the latest result
def save_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=save_result,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize the recognizer
with GestureRecognizer.create_from_options(options) as recognizer:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break
        
        if mvs:

            frame = cv2.undistort(frame, cameraMatrix=cameraMatrix, distCoeffs=distortion_coefficients)

            yuv = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2YUV)
            y, u, v = cv2.split(yuv) # 不知道為啥 v看起來才是水平向量    
            gray_image = cv2.cvtColor(y.copy(), cv2.COLOR_GRAY2BGR)
            cv2.imshow("yuv", gray_image)

        # Create a MediaPipe Image from the frame (ensure correct format)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        

        # The gesture recognizer processes the frame automatically in LIVE_STREAM mode
        # Get the current timestamp in milliseconds
        current_timestamp_ms = int(time.time() * 1000)

        try:
            recognizer.recognize_async(mp_image, current_timestamp_ms)
        except ValueError as e:
            print(f"Error during recognition: {e}")

        # Draw the results on the frame if available
        if latest_result is not None and latest_result.gestures and latest_result.hand_landmarks:
            # Draw hand landmarks
            for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in hand_landmarks
                ])
                
                mp_drawing.draw_landmarks(
                    frame,  # Draw on BGR frame for display
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get gesture information
                if latest_result.gestures and len(latest_result.gestures) > idx:
                    gesture = latest_result.gestures[idx]
                    if gesture:
                        # Display the top gesture
                        top_gesture = gesture[0]
                        gesture_text = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
                        
                        # Calculate position for text (near the wrist)
                        h, w, _ = frame.shape
                        x = int(hand_landmarks[0].x * w)  # Wrist x-coordinate
                        y = int(hand_landmarks[0].y * h)  # Wrist y-coordinate
                        
                        # Draw text with background for better visibility
                        cv2.rectangle(frame, (x-10, y-30), (x + len(gesture_text)*10, y), (0, 0, 0), -1)
                        cv2.putText(frame, gesture_text, (x-5, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the captured frame
        cv2.imshow("Camera", frame)
        video_writer.write(frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        




# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
video_writer.release()
