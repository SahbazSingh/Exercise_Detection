import mediapipe as mp
import cv2
from django.conf import settings
import os

from .bicep_curl import BicepCurlDetection
from .squat import SquatDetection
from .shoulder_lateral import ShoulderLateralRaiseDetection
from .back_pull_down import BackPullDownDetection
from .chest_cable_pull import ChestCablePullDetection

from .utils import rescale_frame

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

EXERCISE_DETECTIONS = None


def load_machine_learning_models():
    """Load all machine learning models"""
    global EXERCISE_DETECTIONS

    if EXERCISE_DETECTIONS is not None:
        return

    print("Loading ML models ...")
    EXERCISE_DETECTIONS = {
        "bicep_curl": BicepCurlDetection(),
        "squat": SquatDetection(),
        "shoulder_lateral_raise": ShoulderLateralRaiseDetection(),
        "back_pull_down": BackPullDownDetection(),
        "chest_cable_pull": ChestCablePullDetection(),
    }


def pose_detection(
    video_file_path: str, video_name_to_save: str, rescale_percent: float = 40
):
    """Pose detection with MediaPipe Pose

    Args:
        video_file_path (str): path to video
        video_name_to_save (str): path to save analyzed video
        rescale_percent (float, optional): Percentage to scale back from the original video size. Defaults to 40.

    """
    cap = cv2.VideoCapture(video_file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * rescale_percent / 100)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * rescale_percent / 100)
    size = (width, height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    save_to_path = f"{settings.MEDIA_ROOT}/{video_name_to_save}"
    out = cv2.VideoWriter(save_to_path, fourcc, fps, size)

    print("PROCESSING VIDEO ...")
    with mp_pose.Pose(
        min_detection_confidence=0.8, min_tracking_confidence=0.8
    ) as pose:
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break

            image = rescale_frame(image, rescale_percent)

            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(244, 117, 66), thickness=2, circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2, circle_radius=1
                ),
            )

            out.write(image)

    print(f"PROCESSED, save to {save_to_path}.")
    return


def exercise_detection(video_file_path: str, video_name_to_save: str, exercise_type: str, rescale_percent: float = 40) -> dict:
    if not video_name_to_save.endswith(".mp4"):
        video_name_to_save += ".mp4"
    
    exercise_detection_instance = EXERCISE_DETECTIONS.get(exercise_type)
    if not exercise_detection_instance:
        raise Exception("Not supported exercise type.")

    cap = cv2.VideoCapture(video_file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * rescale_percent / 100)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * rescale_percent / 100)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    saved_path = os.path.join(settings.MEDIA_ROOT, video_name_to_save)
    out = cv2.VideoWriter(saved_path, fourcc, fps, size)
    
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            
            image_rescaled = rescale_frame(image, rescale_percent)
            image_for_processing = cv2.cvtColor(image_rescaled, cv2.COLOR_BGR2RGB)
            results = pose.process(image_for_processing)
            
            if results.pose_landmarks:
                exercise_detection_instance.detect(mp_results=results, image=image_rescaled, timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            
            out.write(image_rescaled)
    
    cap.release()
    out.release()
    
    result_dict = exercise_detection_instance.handle_detected_results(video_name_to_save)
    exercise_detection_instance.clear_results()

    return {
        "processed_video_path": saved_path,
        **result_dict  # This unpacks the dictionary returned by handle_detected_results
    }


