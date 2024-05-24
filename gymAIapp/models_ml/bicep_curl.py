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
    calculate_angle,
    extract_important_keypoints,
    get_drawing_color,
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class BicepPoseAnalysis:
    def __init__(
        self,
        side: str,
        stage_down_threshold: float,
        stage_up_threshold: float,
        peak_contraction_threshold: float,
        loose_upper_arm_angle_threshold: float,
        visibility_threshold: float,
    ):
        # Initialize thresholds
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {
            "LOOSE_UPPER_ARM": 0,
            "PEAK_CONTRACTION": 0,
        }

        # Params for loose upper arm error detection
        self.loose_upper_arm = False

        # Params for peak contraction error detection
        self.peak_contraction_angle = 1000


    def get_joints(self, landmarks) -> bool:
        """
        Check for joints' visibility then get joints coordinate
        """
        side = self.side.upper()

        # Check visibility
        joints_visibility = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility,
        ]

        is_visible = all([vis > self.visibility_threshold for vis in joints_visibility])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible

        # Get joints' coordinates
        self.shoulder = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y,
        ]
        self.elbow = [
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y,
        ]
        self.wrist = [
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y,
        ]

        return self.is_visible

    def analyze_pose(
        self,
        landmarks,
        frame,
        results,
        timestamp: int,
        lean_back_error: bool = False,
    ):
        """Analyze angles of an arm for error detection

        Args:
            landmarks (): MediaPipe Pose landmarks
            frame (): OpenCV frame
            results (): MediaPipe Pose results
            timestamp (int): timestamp of the frame
            lean_back_error (bool, optional): If there is an lean back error detected, ignore the analysis. Defaults to False.

        Returns:
            _type_: _description_
        """
        has_error = False

        self.get_joints(landmarks)

        # Cancel calculation if visibility is poor
        if not self.is_visible:
            return (None, None, has_error)

        # * Calculate curl angle for counter
        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if bicep_curl_angle > self.stage_down_threshold:
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        # * Calculate the angle between the upper arm (shoulder & joint) and the Y axis
        shoulder_projection = [
            self.shoulder[0],
            1,
        ]  # Represent the projection of the shoulder to the X axis
        ground_upper_arm_angle = int(
            calculate_angle(self.elbow, self.shoulder, shoulder_projection)
        )

        # Stop analysis if lean back error is occur
        if lean_back_error:
            return (bicep_curl_angle, ground_upper_arm_angle, has_error)


        # * Evaluation for LOOSE UPPER ARM error

        
        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
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
                "LOOSE UPPER ARM",
                (0, 70),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Limit the saved frame
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
                results.append(
                    {"stage": "loose upper arm", "frame": frame, "timestamp": timestamp}
                )
        else:
            self.loose_upper_arm = False

        # * Evaluate PEAK CONTRACTION error
        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            # Save peaked contraction every rep
            self.peak_contraction_angle = bicep_curl_angle

        elif self.stage == "down":
            # * Evaluate if the peak is higher than the threshold if True, marked as an error then saved that frame
            if (
                self.peak_contraction_angle != 1000
                and self.peak_contraction_angle >= self.peak_contraction_threshold
            ):

               # Drawing the rectangle starting 40 pixels lower from the top-left corner
                cv2.rectangle(frame, (0, 40), (450, 80), (245, 117, 16), -1)

                # Adjusting the position of the text "ARM ERROR"
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

                # Adjusting the position of the text "WEAK PEAK CONTRACTION"
                cv2.putText(
                    frame,
                    "WEAK PEAK CONTRACTION",
                    (0, 70),  
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )






                self.detected_errors["PEAK_CONTRACTION"] += 1
                results.append(
                    {
                        "stage": "peak contraction",
                        "frame": frame,
                        "timestamp": timestamp,
                    }
                )
                has_error = True

            # Reset params
            self.peak_contraction_angle = 1000

        return (bicep_curl_angle, ground_upper_arm_angle, has_error)

    def get_counter(self) -> int:
        return self.counter

    def reset(self):
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {
            "LOOSE_UPPER_ARM": 0,
            "PEAK_CONTRACTION": 0,
        }

        # Params for loose upper arm error detection
        self.loose_upper_arm = False

        # Params for peak contraction error detection
        self.peak_contraction_angle = 1000


class BicepCurlDetection:
    def __init__(self) -> None:
        # path to the models 
        self.ML_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'bicep_model.pkl')
        self.INPUT_SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'input_scaler.pkl')

        print(f"ML Model Path: {self.ML_MODEL_PATH}")  # Print the ML model path
        print(f"Input Scaler Path: {self.INPUT_SCALER_PATH}")  # Print the input scaler path

        
        # Initialize the model and scaler to None; they will be loaded later
        self.model = None
        self.input_scaler = None

        self.POSTURE_ERROR_THRESHOLD = 0.8
        
        # Call the method to load the machine learning model and scaler
        self.load_machine_learning_model()

        # Initialize analyses for both arms
        self.left_arm_analysis = BicepPoseAnalysis(
            side="left",
            stage_down_threshold=120,
            stage_up_threshold=90,
            peak_contraction_threshold=60,
            loose_upper_arm_angle_threshold=40,
            visibility_threshold=0.65,
        )

        self.right_arm_analysis = BicepPoseAnalysis(
            side="right",
            stage_down_threshold=120,
            stage_up_threshold=90,
            peak_contraction_threshold=60,
            loose_upper_arm_angle_threshold=40,
            visibility_threshold=0.65,
        )

        self.init_important_landmarks()

        self.stand_posture = 0
       
        self.results = []
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

            # Fetch the detailed error description from the 'stage' or a similar detailed key in the 'error' dict
            error_type = error.get("stage", "Unknown error")  # Use 'stage' to store detailed error description like 'Elbow not fully extended'
            timestamp = error.get("timestamp", "Unknown timestamp")

            error_frames_info.append({
                "type": error_type,  # Contains detailed error description
                "timestamp": round(timestamp, 1),
                "frame": frame_path  # or None if saving failed
            })

        total_curls = self.left_arm_analysis.get_counter() + self.right_arm_analysis.get_counter()

        return {
            "total_curls": total_curls,
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


    def detect(
        self,
        mp_results,
        image,
        timestamp: int,
    ) -> None:
        """Error detection

        Args:
            mp_results (): MediaPipe results
            image (): OpenCV image
            timestamp (int): Current time of the frame
        """
        self.has_error = False

        try:
            video_dimensions = [image.shape[1], image.shape[0]]
            landmarks = mp_results.pose_landmarks.landmark

            # * Model prediction for Lean-back error
            # Extract keypoints from frame for the input
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

            # Print the predicted class and its probability
            #print(f"Predicted Class: {predicted_class}, Probability: {class_prediction_probability}")

            current_time = time.time()

            if class_prediction_probability >= self.POSTURE_ERROR_THRESHOLD and (current_time - self.last_posture_error_time > 1):
                self.stand_posture = predicted_class
                self.last_posture_error_time = current_time

                if self.stand_posture == 1:
                    self.results.append(
                        {
                            "stage": "lean too far back",
                            "frame": image,
                            "timestamp": timestamp,
                        }
                    )

                self.has_error = True

            

            # * Arms analysis for errors
            # Left arm
            (
                left_bicep_curl_angle,
                left_ground_upper_arm_angle,
                left_arm_error,
            ) = self.left_arm_analysis.analyze_pose(
                landmarks=landmarks,
                frame=image,
                results=self.results,
                timestamp=timestamp,
                lean_back_error=(self.stand_posture == 1),
            )

            # Right arm
            (
                right_bicep_curl_angle,
                right_ground_upper_arm_angle,
                right_arm_error,
            ) = self.right_arm_analysis.analyze_pose(
                landmarks=landmarks,
                frame=image,
                results=self.results,
                timestamp=timestamp,
                lean_back_error=(self.stand_posture == 1),
            )

            self.has_error = (
                True if (right_arm_error or left_arm_error) else self.has_error
            )

            # Visualization
            # Draw landmarks and connections
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
                str("LEAN TOO FAR BACK" if self.stand_posture == 1 else "CORRECT POSTURE"),
                (240, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # * Visualize angles
            # Visualize LEFT arm calculated angles
            if self.left_arm_analysis.is_visible:
                cv2.putText(
                    image,
                    str(left_bicep_curl_angle),
                    tuple(
                        np.multiply(
                            self.left_arm_analysis.elbow, video_dimensions
                        ).astype(int)
                    ),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(left_ground_upper_arm_angle),
                    tuple(
                        np.multiply(
                            self.left_arm_analysis.shoulder, video_dimensions
                        ).astype(int)
                    ),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Visualize RIGHT arm calculated angles
            if self.right_arm_analysis.is_visible:
                cv2.putText(
                    image,
                    str(right_bicep_curl_angle),
                    tuple(
                        np.multiply(
                            self.right_arm_analysis.elbow, video_dimensions
                        ).astype(int)
                    ),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(right_ground_upper_arm_angle),
                    tuple(
                        np.multiply(
                            self.right_arm_analysis.shoulder, video_dimensions
                        ).astype(int)
                    ),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        except Exception as e:
            traceback.print_exc()
            raise e
        













    
   












    
   