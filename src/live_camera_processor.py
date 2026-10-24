"""
Live Camera Image & Stream Processing Utilities for DeepShield
Implements real-time biometric face detection, landmark extraction, cybernetic HUD visualization,
and spatial EfficientNet-B0 inference on live camera frames.
"""

import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LiveCameraProcessor:
    def __init__(self, task_model_path="models/face_landmarker.task"):
        self.task_model_path = task_model_path
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """Initializes MediaPipe FaceLandmarker for biometric tracking."""
        if os.path.exists(self.task_model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=self.task_model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                print("FaceLandmarker initialization in LiveCameraProcessor failed:", e)
                self.detector = None

    def process_camera_snapshot(self, image_input, visual_detector):
        """
        Processes a single live camera snapshot:
        1. Normalizes image to RGB numpy array.
        2. Detects facial region and extracts biometric landmarks.
        3. Draws futuristic cyber HUD overlay with bounding box and landmark mesh.
        4. Crops face ROI and executes EfficientNet-B0 deepfake detection inference.
        """
        # Convert input to RGB PIL Image and OpenCV numpy array
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif hasattr(image_input, "read") or hasattr(image_input, "getvalue"):
            # Streamlit UploadedFile or BytesIO
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        img_np = np.array(pil_img)
        h, w, _ = img_np.shape
        annotated_np = img_np.copy()

        face_detected = False
        face_bbox = None
        landmarks_count = 0
        face_crop_pil = pil_img

        # 1. MediaPipe Biometric Detection
        if self.detector is not None:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                detection_result = self.detector.detect(mp_image)

                if detection_result and detection_result.face_landmarks:
                    face_detected = True
                    face_landmarks = detection_result.face_landmarks[0]
                    landmarks_count = len(face_landmarks)

                    xs = [lm.x * w for lm in face_landmarks]
                    ys = [lm.y * h for lm in face_landmarks]

                    x_min, x_max = max(0, int(min(xs))), min(w - 1, int(max(xs)))
                    y_min, y_max = max(0, int(min(ys))), min(h - 1, int(max(ys)))

                    # Add padding around face
                    box_w = x_max - x_min
                    box_h = y_max - y_min
                    pad_x = int(box_w * 0.15)
                    pad_y = int(box_h * 0.20)

                    crop_x1 = max(0, x_min - pad_x)
                    crop_y1 = max(0, y_min - pad_y)
                    crop_x2 = min(w, x_max + pad_x)
                    crop_y2 = min(h, y_max + pad_y)

                    face_bbox = {
                        "x": crop_x1,
                        "y": crop_y1,
                        "w": crop_x2 - crop_x1,
                        "h": crop_y2 - crop_y1
                    }

                    # Crop high-res face
                    crop_np = img_np[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop_np.size > 0:
                        face_crop_pil = Image.fromarray(crop_np)

                    # Draw Cyber Biometric Landmark Mesh Overlay
                    # Draw subtle cyan dots for key landmarks
                    for i, lm in enumerate(face_landmarks):
                        # Draw subset of landmark dots for sleek visualization
                        if i % 3 == 0:
                            lx, ly = int(lm.x * w), int(lm.y * h)
                            cv2.circle(annotated_np, (lx, ly), 1, (56, 189, 248), -1)

                    # Highlight upper and lower lip landmarks (13 and 14)
                    if len(face_landmarks) > 14:
                        u_lip = face_landmarks[13]
                        l_lip = face_landmarks[14]
                        cv2.circle(annotated_np, (int(u_lip.x * w), int(u_lip.y * h)), 3, (236, 72, 153), -1)
                        cv2.circle(annotated_np, (int(l_lip.x * w), int(l_lip.y * h)), 3, (236, 72, 153), -1)
                        cv2.line(
                            annotated_np,
                            (int(u_lip.x * w), int(u_lip.y * h)),
                            (int(l_lip.x * w), int(l_lip.y * h)),
                            (236, 72, 153),
                            1
                        )

                    # Draw Cyber Corner Brackets around Face
                    self._draw_cyber_brackets(annotated_np, crop_x1, crop_y1, crop_x2, crop_y2, color=(56, 189, 248))

            except Exception as e:
                print("Error in MediaPipe processing:", e)

        # Fallback to OpenCV Haar Cascade if MediaPipe failed or found no face
        if not face_detected:
            face_detected, face_bbox, face_crop_pil = self._fallback_haar_detect(img_np, annotated_np)

        # 2. Run Spatial EfficientNet-B0 Deepfake Detection
        inference_result = None
        if visual_detector is not None:
            # Send face crop to model
            inference_result = visual_detector.predict_frames([(0, face_crop_pil)])

        annotated_pil = Image.fromarray(annotated_np)

        return {
            "face_detected": face_detected,
            "landmarks_count": landmarks_count,
            "face_bbox": face_bbox,
            "annotated_image": annotated_pil,
            "face_crop": face_crop_pil,
            "inference": inference_result
        }

    def _draw_cyber_brackets(self, img, x1, y1, x2, y2, color=(56, 189, 248), thickness=2, arm_len=18):
        """Draws high-tech sci-fi corner brackets around detected face."""
        # Top-left
        cv2.line(img, (x1, y1), (x1 + arm_len, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + arm_len), color, thickness)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - arm_len, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + arm_len), color, thickness)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + arm_len, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - arm_len), color, thickness)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - arm_len, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - arm_len), color, thickness)

        # Subtle target tag
        cv2.putText(
            img,
            "BIOMETRIC TARGET ACQUIRED",
            (x1 + 4, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA
        )

    def _fallback_haar_detect(self, img_np, annotated_np):
        """Fallback face detection using OpenCV Haar Cascade."""
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if not os.path.exists(cascade_path):
                return False, None, Image.fromarray(img_np)

            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                self._draw_cyber_brackets(annotated_np, x, y, x + w, y + h, color=(52, 211, 153))
                face_crop = img_np[y:y + h, x:x + w]
                return True, {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}, Image.fromarray(face_crop)
        except Exception:
            pass

        return False, None, Image.fromarray(img_np)

    @staticmethod
    def probe_local_cameras(max_probe=2):
        """Probes local system for available OpenCV camera devices."""
        available = []
        for idx in range(max_probe):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append({
                            "index": idx,
                            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            "fps": round(cap.get(cv2.CAP_PROP_FPS), 1)
                        })
                    cap.release()
            except Exception:
                pass
        return available
