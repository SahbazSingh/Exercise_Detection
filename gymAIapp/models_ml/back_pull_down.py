import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import traceback
import os
from django.conf import settings
import time

from .utils import (
    extract_important_keypoints,
    get_drawing_color, 
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class BackPullDownAnalysis:
    def __init__(self, contraction_threshold: float, extension_threshold: float, visibility_threshold: float):
        self.contraction_threshold = contraction_threshold  # Threshold for contraction phase
        self.extension_threshold = extension_threshold  # Threshold for extension phase
        self.visibility_threshold = visibility_threshold  # Visibility threshold for landmarks
        
        self.counter = 0
        self.mid_point_reached = False  # Tracks if the midpoint of the movement has been reached
        

    def analyze_pose(self, landmarks):
        
        left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        
        # Calculate distances between wrists and hips for both sides
        left_distance = np.linalg.norm(left_wrist - left_hip)
        right_distance = np.linalg.norm(right_wrist - right_hip)
        avg_distance = (left_distance + right_distance) / 2

        # Detect contraction (pull-down phase)
        if avg_distance < self.contraction_threshold and not self.mid_point_reached:
            self.mid_point_reached = True

        # Detect extension (return phase)
        if avg_distance > self.extension_threshold and self.mid_point_reached:
            self.counter += 1  # Increment rep count
            self.mid_point_reached = False

        return self.counter, self.mid_point_reached

    def get_counter(self) -> int:
        return self.counter

    def reset(self):
        self.counter = 0  # Reset repetition counter
        self.stage = "initial"  # Reset the stage to some initial value, such as "initial"
        self.is_visible = True  # Reset visibility flag
        # Reset the dictionary that tracks detected errors; adjust according to your actual error tracking
        self.detected_errors = {
            "TOO_HIGH": 0,
            
        }
        self.too_high_flag = False  # Reset flag for specific error detection

class BackPullDownDetection:
    def __init__(self) -> None:
        self.ML_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'back_pull_down_model.pkl')
        self.INPUT_SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'pull_down_input_scaler.pkl')

        self.model = None
        self.input_scaler = None  

        # Load machine learning model and scaler
        self.load_machine_learning_model()

        # instantiation with all required arguments
        self.back_pull_down_analysis = BackPullDownAnalysis(
            contraction_threshold=0.15,  
            extension_threshold=0.20, 
            visibility_threshold=0.5  
        )

        self.init_important_landmarks()
        self.results = []
        self.POSTURE_ERROR_THRESHOLD = 0.8
        self.has_error = False
        self.last_posture_error_time = 0 


    def init_important_landmarks(self):
        self.important_landmarks = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW",
            "LEFT_ELBOW", "RIGHT_WRIST", "LEFT_WRIST", "LEFT_HIP", "RIGHT_HIP",
        ]
        self.headers = ["label"]
        for lm in self.important_landmarks:
            self.headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
        

    def load_machine_learning_model(self):
        """Load machine learning model and input scaler."""
        try:
            with open(self.ML_MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.INPUT_SCALER_PATH, 'rb') as f2:
                self.input_scaler = pickle.load(f2)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {e.filename}") from e
        except Exception as e:
            raise Exception(f"Error loading machine learning model: {e}") from e

    def save_error_frame(self, frame, file_name, index):
        """Save frame as evidence and return the relative path."""
        # Create the error_frames directory if it doesn't exist
        error_frames_dir = os.path.join(settings.MEDIA_ROOT, 'error_frames')
        os.makedirs(error_frames_dir, exist_ok=True)

        # Generate a filename for the frame
        frame_filename = f"{file_name}_{index}.jpg"
        frame_save_path = os.path.join(error_frames_dir, frame_filename)

        # Save the frame
        success = cv2.imwrite(frame_save_path, frame)
        if success:
            # Return the relative path to be saved in the database
            return os.path.join('error_frames', frame_filename)
        else:
            return None

    def handle_detected_results(self, video_name: str) -> dict:
        """Handle and summarize detected errors and repetitions."""
        file_name, _ = os.path.splitext(video_name)
        error_frames_info = []

        for index, error in enumerate(self.results):
            frame_path = self.save_error_frame(error["frame"], file_name, index)

            # Fetch the detailed error description from the 'stage' key or default to 'type'
            error_type = error.get("stage", error.get("type", "Unknown error"))
            timestamp = error.get("timestamp", "Unknown timestamp")

            error_frames_info.append({
                "type": error_type,  # Contains detailed error description
                "timestamp": round(timestamp, 1),
                "frame": frame_path  # This might be None if saving failed
            })

        total_reps = self.back_pull_down_analysis.get_counter()  

        return {
            "total_reps": total_reps,
            "error_frames_info": error_frames_info,
            "feedback": "Analyze the detected errors for detailed feedback."
        }

    def clear_results(self) -> None:
        """Clear results for a new detection session."""
        self.results = []
        self.has_error = False
        self.back_pull_down_analysis.reset()  
        self.last_posture_error_time = 0


    def detect(self, mp_results, image, timestamp: int) -> None:
        try:
            if mp_results.pose_landmarks:
                landmarks = mp_results.pose_landmarks.landmark

                # Extract important keypoints for the model
                row = extract_important_keypoints(mp_results, self.important_landmarks)
                X = pd.DataFrame([row], columns=self.headers[1:])
                X_scaled = self.input_scaler.transform(X)

                # Predict using the machine learning model
                prediction = self.model.predict(X_scaled)  
                if prediction.ndim == 2:
                    predicted_class = np.argmax(prediction[0], axis=0)  # Adjusted to axis 0 if already accessing first element
                    prediction_probability = round(max(prediction[0]), 2)
                else:
                    predicted_class = np.argmax(prediction, axis=0)  # No axis needed if prediction is genuinely 1D
                    prediction_probability = round(max(prediction), 2)

                # Drawing a rectangle for displaying all related information starting from the top left corner
                cv2.rectangle(image, (0, 0), (250, 100), (245, 117, 16), -1)  

                # Analyze the pull-down exercise posture
                counter, mid_point_reached = self.back_pull_down_analysis.analyze_pose(landmarks)
                feedback_msg = "Extend" if mid_point_reached else "Pull Down"
                feedback_msg_color = (0, 0, 255) if mid_point_reached else (0, 255, 0)

                # Display dynamic feedback message based on analysis
                cv2.putText(image, feedback_msg, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, feedback_msg_color, 1, cv2.LINE_AA)

                # Display counter of repetitions
                cv2.putText(image, f"Count: {counter}", (10, 55),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Determine if there's a lean back error based on the prediction and threshold
                current_time = time.time()

                if prediction_probability >= self.POSTURE_ERROR_THRESHOLD and (current_time - self.last_posture_error_time > 1):
                    lean_back_error = predicted_class
                    self.last_posture_error_time = current_time

                    if lean_back_error == 1:
                        cv2.putText(image, "Lean Back Error Detected", (10, 85), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        self.results.append(
                            {
                                "stage": "lean too far back",
                                "frame": image,
                                "timestamp": timestamp,
                            }
                        )

                        self.has_error = True

                landmark_color, connection_color = get_drawing_color(self.has_error)
                mp_drawing.draw_landmarks(
                    image,
                    mp_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=landmark_color, thickness=2, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(
                        color=connection_color, thickness=2, circle_radius=1
                    ),
                )
                
        except Exception as e:
            traceback.print_exc()
            print(f"Error in detect function: {e}")


