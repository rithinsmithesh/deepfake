"""
Visual Analysis Module for DeepShield
Implements EfficientNet-B0 frame extraction, preprocessing, and inference.
Follows exact configuration from deepshield-model.ipynb.
"""

import os
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms


# Kaggle verified benchmark metrics from deepshield-model.ipynb (Cell 20)
VERIFIED_BENCHMARKS = {
    "architecture": "EfficientNet-B0",
    "dataset": "FakeAVCeleb v1.2",
    "classes": {0: "FAKE", 1: "REAL"},
    "test_frames": 3234,
    "accuracy": 0.9975,
    "precision": 0.9735,
    "recall": 0.9735,
    "f1_score": 0.9735,
    "confusion_matrix": {
        "fake_correct": 3079,
        "fake_misclassified_as_real": 4,
        "real_misclassified_as_fake": 4,
        "real_correct": 147,
    },
    "training_epochs": 30,
    "optimizer": "Adam (lr=0.0001)",
    "loss": "CrossEntropyLoss"
}


class DeepShieldVisualDetector:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.checkpoint_path = checkpoint_path or "models/FINAL_EfficientNet_B0_FakeAVCeleb.pth"
        self.model = None
        self.is_loaded = False
        self.load_error = None
        self.classes = {0: "FAKE", 1: "REAL"}

        # Transform exactly as defined in deepshield-model.ipynb (Cell 13)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.attempt_load()

    def attempt_load(self, custom_path=None):
        """Attempts to load model weights if the checkpoint exists."""
        target_path = custom_path or self.checkpoint_path

        if not os.path.exists(target_path):
            self.is_loaded = False
            self.load_error = f"Checkpoint file not found at: {target_path}"
            return False

        try:
            # Instantiate EfficientNet-B0 architecture
            model = efficientnet_b0(weights=None)
            # Replace classifier head for 2 classes (0=FAKE, 1=REAL)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

            state_dict = torch.load(target_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.model = model
            self.is_loaded = True
            self.checkpoint_path = target_path
            self.load_error = None
            return True
        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Error loading checkpoint: {str(e)}"
            return False

    def extract_video_metadata(self, video_path):
        """Extracts technical metadata from video container."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        cap.release()
        return {
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration": round(duration, 2)
        }

    def sample_frames(self, video_path, num_samples=16):
        """Uniformly samples representative frames across video duration."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        # Fast path: If header reports total_frames > 0, sample directly
        if total_frames > 0:
            actual_samples = min(num_samples, total_frames)
            sample_indices = np.linspace(0, total_frames - 1, actual_samples, dtype=int)
            target_idx_set = set(sample_indices)
            frames = []
            timestamps = []
            current_idx = 0
            while cap.isOpened() and len(frames) < actual_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_idx in target_idx_set:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    timestamp = current_idx / fps
                    frames.append((current_idx, pil_img))
                    timestamps.append(timestamp)
                current_idx += 1
            cap.release()
            if len(frames) > 0:
                return frames, timestamps

        # Resilient path for dynamic/live streams: read sequentially until EOF
        all_frames = []
        all_times = []
        idx = 0
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append((idx, Image.fromarray(rgb)))
            all_times.append(idx / fps)
            idx += 1
        cap.release()

        if len(all_frames) == 0:
            return [], []

        if len(all_frames) <= num_samples:
            return all_frames, all_times

        sample_idx = np.linspace(0, len(all_frames) - 1, num_samples, dtype=int)
        return [all_frames[i] for i in sample_idx], [all_times[i] for i in sample_idx]

    @staticmethod
    def _coerce_image(image):
        """Convert WebRTC/OpenCV frames to the PIL format expected by torchvision."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.ndim == 2:
                return Image.fromarray(array).convert("RGB")
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise TypeError(f"Unsupported image shape: {array.shape}")
            if array.shape[2] == 4:
                array = array[:, :, :3]
            return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    def predict_frames(self, pil_frames):
        """
        Runs genuine PyTorch inference if checkpoint is loaded.
        NEVER invents predictions if checkpoint is missing.
        """
        if not self.is_loaded or self.model is None:
            return {
                "status": "missing_checkpoint",
                "message": (
                    "Visual model checkpoint ('FINAL_EfficientNet_B0_FakeAVCeleb.pth') is not loaded. "
                    "In accordance with rigorous scientific integrity, simulated predictions are not generated. "
                    "Place your trained checkpoint in models/ or upload it via the settings."
                ),
                "device": str(self.device),
                "benchmarks": VERIFIED_BENCHMARKS
            }

        if not pil_frames or len(pil_frames) == 0:
            return {
                "status": "error",
                "message": "No decodable frames were found in the video stream.",
                "overall_prediction": "INCONCLUSIVE",
                "overall_confidence": 0.0,
                "mean_fake_prob": 50.0,
                "mean_real_prob": 50.0,
                "frame_results": [],
                "inference_time_sec": 0.0,
                "frames_analyzed": 0,
                "device": str(self.device),
                "benchmarks": VERIFIED_BENCHMARKS
            }

        start_time = time.time()
        normalized_frames = [(f_idx, self._coerce_image(img)) for f_idx, img in pil_frames]
        tensors = []
        for _, img in normalized_frames:
            tensor = self.transform(img)
            tensors.append(tensor)

        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        inference_time = round(time.time() - start_time, 4)

        frame_results = []
        for i, (f_idx, _) in enumerate(normalized_frames):
            frame_results.append({
                "frame_index": f_idx,
                "label": self.classes[int(preds[i])],
                "pred_id": int(preds[i]),
                "prob_fake": float(probs[i][0]),
                "prob_real": float(probs[i][1]),
            })

        mean_fake_prob = float(np.mean(probs[:, 0]))
        mean_real_prob = float(np.mean(probs[:, 1]))
        overall_pred = "FAKE" if mean_fake_prob >= 0.50 else "REAL"
        confidence = mean_fake_prob if overall_pred == "FAKE" else mean_real_prob

        return {
            "status": "success",
            "overall_prediction": overall_pred,
            "overall_confidence": round(confidence * 100, 2),
            "mean_fake_prob": round(mean_fake_prob * 100, 2),
            "mean_real_prob": round(mean_real_prob * 100, 2),
            "frame_results": frame_results,
            "inference_time_sec": inference_time,
            "frames_analyzed": len(normalized_frames),
            "device": str(self.device),
            "benchmarks": VERIFIED_BENCHMARKS
        }
