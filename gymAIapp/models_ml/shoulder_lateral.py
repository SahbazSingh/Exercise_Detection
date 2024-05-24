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

class ShoulderLateralRaiseAnalysis:
    def __init__(self, side: str, raise_threshold: float, lower_threshold: float, visibility_threshold: float):
        self.raise_threshold = raise_threshold
        self.lower_threshold = lower_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "lower"
        self.is_visible = True
        self.detected_errors = {
            "TOO_HIGH": 0,
        }
        self.too_high_flag = False

    def get_joints(self, landmarks) -> bool:
        side = self.side.upper()

        joints_visibility = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility,
        ]

        is_visible = all(vis > self.visibility_threshold for vis in joints_visibility)
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible

        self.shoulder = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y,
        ]
        self.hip = [
            landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].y,
        ]
        self.wrist = [
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y,
        ]

        return self.is_visible

    def analyze_pose(self, landmarks, frame, results, timestamp: int):
        self.get_joints(landmarks)
        has_error = False

        if not self.is_visible:
            return (None, None, has_error)

        
        shoulder_to_wrist_height = self.wrist[1] - self.shoulder[1]

        # Determine the current stage of the exercise
        if shoulder_to_wrist_height < self.raise_threshold and self.stage == "raise":
            self.stage = "lower"
            self.too_high_flag = False  # Reset flag when transitioning to lower stage
        elif shoulder_to_wrist_height >= self.raise_threshold and self.stage == "lower":
            self.stage = "raise"
            self.counter += 1

        # Evaluation for TOO_HIGH error - only count once per rep
        if shoulder_to_wrist_height < self.lower_threshold and not self.too_high_flag:
            has_error = True

            cv2.rectangle(frame, (0, 40), (450, 80), (245, 117, 16), -1)
            cv2.putText(
                frame,
                "ARM ERROR",
                (0, 52),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "TOO HIGH",
                (0, 70),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            self.detected_errors["TOO_HIGH"] += 1
            self.too_high_flag = True  # Prevent multiple counts in the same rep
            error_type = "TOO_HIGH"
            results.append({"stage": "too high", "frame": frame, "timestamp": timestamp, "type": error_type})

        return (shoulder_to_wrist_height, self.stage, has_error)

    def get_counter(self) -> int:
        return self.counter

    def reset(self):
        self.counter = 0
        self.stage = "lower"
        self.is_visible = True
        self.detected_errors = {
            "TOO_HIGH": 0,
        }
        self.too_high_flag = False


class ShoulderLateralRaiseDetection:
    def __init__(self) -> None:
        # Paths to the 'ml_models' directory at the project root
        self.ML_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'Lateral_Raise_model.pkl')
        self.INPUT_SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'shoulder_input_scaler.pkl')

        print(f"ML Model Path: {self.ML_MODEL_PATH}")
        print(f"Input Scaler Path: {self.INPUT_SCALER_PATH}")

        # Initialize the model and scaler to None; they will be loaded later
        self.model = None
        self.input_scaler = None

        self.POSTURE_ERROR_THRESHOLD = 0.8
        
        # Call the method to load the machine learning model and scaler
        self.load_machine_learning_model()

        # Initialize analyses for both arms
        self.left_arm_analysis = ShoulderLateralRaiseAnalysis(
            side="left",
            raise_threshold=0.12, 
            lower_threshold=-0.10,  
            visibility_threshold=0.65,
        )

        self.right_arm_analysis = ShoulderLateralRaiseAnalysis(
            side="right",
            raise_threshold=0.12,  
            lower_threshold=-0.10,  
            visibility_threshold=0.65,
        )

        self.init_important_landmarks()

        self.results = []
        self.stand_posture = 0 

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
        file_name, _ = os.path.splitext(video_name)
        save_folder = os.path.join(settings.MEDIA_ROOT, "error_frames")
        os.makedirs(save_folder, exist_ok=True)  # Ensure the directory exists

        error_frames_info = []  # This will store information about each error
        for index, error in enumerate(self.results):
            frame_path = None
            try:
                frame_path = os.path.join(save_folder, f"{file_name}_{index}.jpg")
                cv2.imwrite(frame_path, error["frame"])
                frame_path = os.path.relpath(frame_path, settings.MEDIA_ROOT)  # Convert to relative path if needed
            except Exception as e:
                print(f"ERROR cannot save frame: {e}")

            # Collect detailed error information from the 'stage' key or use 'type' if 'stage' is not available
            error_type = error.get("stage", "Unknown error")  
            timestamp = error.get("timestamp", "Unknown timestamp")

            error_frames_info.append({
                "type": error_type,  # Contains detailed error description
                "timestamp": round(timestamp, 1),
                "frame": frame_path  # or None if saving failed
            })

        total_reps = max(self.left_arm_analysis.get_counter(), self.right_arm_analysis.get_counter())

        return {
            "total_reps": total_reps,
            "error_frames_info": error_frames_info,
            "feedback": "Analyze the detected errors for detailed feedback."
            
        }



    def clear_results(self) -> None:
        self.stand_posture = 0
       
        self.results = []
        self.has_error = False

        self.right_arm_analysis.reset()
        self.left_arm_analysis.reset()

        self.last_posture_error_time = 0

    
    def detect(self, mp_results, image, timestamp: int) -> None:
        """Detect shoulder lateral raises and errors in posture or execution."""
        self.has_error = False

        try:
            if mp_results.pose_landmarks:
                landmarks = mp_results.pose_landmarks.landmark

                row = extract_important_keypoints(mp_results, self.important_landmarks)
                X = pd.DataFrame(
                    [
                        row,
                    ],
                    columns=self.headers[1:],
                )
                X = pd.DataFrame(self.input_scaler.transform(X))

                # Make prediction and its probability
                predicted_class = self.model.predict(X)[0]
                prediction_probabilities = self.model.predict_proba(X)[0]
                class_prediction_probability = round(
                    prediction_probabilities[np.argmax(prediction_probabilities)], 2
                )

                current_time = time.time()

                # Evaluate the prediction for potential action
                if class_prediction_probability >= self.POSTURE_ERROR_THRESHOLD and (current_time - self.last_posture_error_time > 1):
                    self.stand_posture = predicted_class
                    self.last_posture_error_time = current_time

                    if self.stand_posture == 1:

                        self.results.append(
                            {
                                "stage": "Back too straight",
                                "frame": image,
                                "timestamp": timestamp,
                            }
                        )

                        self.has_error = True

                #Analysis for both arms
                left_shoulder_to_wrist_height, left_stage, left_has_error = self.left_arm_analysis.analyze_pose(
                    landmarks=landmarks,
                    frame=image,
                    results=self.results,
                    timestamp=timestamp
                )

                right_shoulder_to_wrist_height, right_stage, right_has_error = self.right_arm_analysis.analyze_pose(
                    landmarks=landmarks,
                    frame=image,
                    results=self.results,
                    timestamp=timestamp
                )

                self.has_error = (
                    True if (left_has_error or right_has_error) else self.has_error
                )


                # Drawing landmarks and connections
                landmark_color, connection_color = get_drawing_color(self.has_error)
                mp_drawing.draw_landmarks(
                    image,
                    mp_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1)
                )


                
                # Status box
                cv2.rectangle(image, (0, 0), (450, 40), (245, 117, 16), -1)

                # Display probability
                cv2.putText(
                    image,
                    "RIGHT REPS",
                    (0, 12),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(self.right_arm_analysis.counter)
                    if self.right_arm_analysis.is_visible
                    else "unknown",
                    (0, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Display Left Counter
                cv2.putText(
                    image,
                    "LEFT REPS",
                    (120, 12),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(self.left_arm_analysis.counter)
                    if self.left_arm_analysis.is_visible
                    else "unknown",
                    (120, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Lean back error
                cv2.putText(
                    image,
                    "ERROR DETECTION",
                    (240, 12),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                
                cv2.putText(
                    image,
                    str("Back too straight" if self.stand_posture == 1 else "CORRECT POSTURE"),
                    (240, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                
        except Exception as e:
            traceback.print_exc()
            raise e



                

                


                

                
