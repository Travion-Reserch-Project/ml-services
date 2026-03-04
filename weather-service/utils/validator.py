from fastapi import HTTPException
import mediapipe as mp
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

def validate_human_skin(image_np: np.ndarray):
    if not contains_human(image_np):
        raise HTTPException(status_code=400, detail="No human detected in image")

    if not contains_skin(image_np):
        raise HTTPException(status_code=400, detail="Insufficient skin region detected")

def contains_human(image_np: np.ndarray) -> bool:
    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    options = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path="models/blaze_face_short_range.tflite"
        ),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.5
    )

    with FaceDetector.create_from_options(options) as detector:
        result = detector.detect(mp_image)
        return len(result.detections) > 0

def contains_skin(image_np: np.ndarray, threshold: float = 0.03) -> bool:
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.sum(mask > 0) / mask.size

    return skin_ratio > threshold
