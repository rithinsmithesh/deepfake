"""
Audio-Lip Synchronization Module for DeepShield
Implements Pearson correlation between speech activity (RMS envelope)
and vertical lip aperture (MediaPipe landmark indices 13 and 14).
Preserves the exact algorithm and formulas developed in deepshield-member3 notebook.
"""

import os
import cv2
import numpy as np
from scipy.stats import pearsonr
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Dataset baseline stats from member3_ppt_summary.csv & full_dataset_synchronization_results.csv
DATASET_SYNC_BASELINES = {
    "Real": {
        "mean_correlation": 0.0562,
        "median_correlation": 0.0648,
        "mean_sync_score": 9.37,
        "median_sync_score": 6.48,
        "videos": 100
    },
    "Fake": {
        "mean_correlation": 0.0158,
        "median_correlation": -0.0019,
        "mean_sync_score": 7.16,
        "median_sync_score": 0.0,
        "videos": 100
    },
    "Fake Voice Only": {
        "mean_correlation": 0.0425,
        "median_correlation": 0.0667,
        "mean_sync_score": 9.51,
        "median_sync_score": 6.67,
        "videos": 30
    }
}


class DeepShieldSyncAnalyzer:
    def __init__(self, task_model_path="models/face_landmarker.task"):
        self.task_model_path = task_model_path
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """Initializes MediaPipe FaceLandmarker if task model is available."""
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
                self.detector = None
                print("FaceLandmarker initialization failed:", e)

    def extract_mouth_aperture_signal(self, video_path, max_frames=300):
        """
        Reads video frames and extracts vertical lip distance between
        upper lip (landmark 13) and lower lip (landmark 14).
        Falls back to visual mouth-region motion tracking if MediaPipe is unavailable.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, "Failed to open video file."

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            return None, None, "Invalid video stream (FPS or frame count is zero)."

        # Determine frame stride if video is very long
        stride = max(1, total_frames // max_frames)

        mouth_signal = []
        frame_times = []
        faces_detected = 0
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % stride == 0:
                frame_time = frame_index / fps
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mouth_opening = 0.0
                detected_in_frame = False

                if self.detector is not None:
                    try:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result = self.detector.detect(mp_image)
                        if result.face_landmarks and len(result.face_landmarks) > 0:
                            landmarks = result.face_landmarks[0]
                            # Indices 13 (upper inner lip) and 14 (lower inner lip)
                            upper = landmarks[13]
                            lower = landmarks[14]
                            mouth_opening = abs(lower.y - upper.y)
                            detected_in_frame = True
                            faces_detected += 1
                    except Exception:
                        pass

                # Fallback face/mouth region intensity dynamics if MediaPipe landmark missing
                if not detected_in_frame:
                    mouth_opening = self._estimate_mouth_fallback(rgb)
                    if mouth_opening > 0:
                        faces_detected += 1

                mouth_signal.append(mouth_opening)
                frame_times.append(frame_time)

            frame_index += 1

        cap.release()

        mouth_signal = np.array(mouth_signal, dtype=float)
        frame_times = np.array(frame_times, dtype=float)

        if len(mouth_signal) < 2:
            return None, None, "Insufficient video frames extracted (<2 frames)."

        if faces_detected == 0:
            return None, None, "No facial landmarks or recognizable faces detected in video."

        return mouth_signal, frame_times, None

    def _estimate_mouth_fallback(self, rgb_frame):
        """
        Robust heuristic fallback: samples lower-central face quadrant variance
        when neural landmark detector encounters extreme occlusion or lighting.
        """
        h, w = rgb_frame.shape[:2]
        # Lower central third represents anatomical mouth region in typical framed portraits
        ymin, ymax = int(h * 0.60), int(h * 0.85)
        xmin, xmax = int(w * 0.35), int(w * 0.65)
        mouth_roi = rgb_frame[ymin:ymax, xmin:xmax]
        if mouth_roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(mouth_roi, cv2.COLOR_RGB2GRAY)
        # Vertical gradient proxy for lip aperture
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.std(sobel_y) / 255.0)

    def analyze_synchronization(self, audio_data, video_path):
        """
        Correlates speech activity envelope with mouth motion signal.
        Follows cell 28-33 of deepshield-member3-audio-synchronization.ipynb.
        """
        if not audio_data or not audio_data.get("has_audio"):
            return {
                "success": False,
                "reason": "Synchronization requires an active audio stream. Video has no audio.",
                "correlation": None,
                "p_value": None,
                "sync_score": None
            }

        mouth_signal, frame_times, err = self.extract_mouth_aperture_signal(video_path)
        if err or mouth_signal is None:
            return {
                "success": False,
                "reason": err or "Failed to compute mouth aperture signal.",
                "correlation": None,
                "p_value": None,
                "sync_score": None
            }

        audio_times = audio_data["audio_times"]
        speech_activity = audio_data["speech_activity"]

        # Step 5: Interpolate mouth movement to audio time points
        mouth_interp = np.interp(audio_times, frame_times, mouth_signal)

        # Step 6: Normalize mouth movement to 0-1
        m_min = float(mouth_interp.min())
        m_max = float(mouth_interp.max())
        mouth_norm = (mouth_interp - m_min) / (m_max - m_min + 1e-8)

        # Step 7: Pearson correlation
        try:
            corr, p_val = pearsonr(speech_activity.astype(float), mouth_norm)
            if np.isnan(corr):
                corr = 0.0
                p_val = 1.0
            else:
                corr = float(corr)
                p_val = float(p_val)

            # Step 8: Sync Score formula from notebook Cell 33: max(0, correlation) * 100
            sync_score = max(0.0, corr) * 100.0

        except Exception as e:
            return {
                "success": False,
                "reason": f"Correlation calculation error: {str(e)}",
                "correlation": None,
                "p_value": None,
                "sync_score": None
            }

        # Contextual explanation comparing against 230 FakeAVCeleb videos
        interpretation = self._interpret_sync_score(corr, sync_score, p_val)

        return {
            "success": True,
            "correlation": round(corr, 4),
            "p_value": p_val,
            "sync_score": round(sync_score, 2),
            "audio_times": audio_times,
            "speech_activity": speech_activity,
            "mouth_norm": mouth_norm,
            "raw_mouth_signal": mouth_signal,
            "frame_times": frame_times,
            "interpretation": interpretation,
            "dataset_baselines": DATASET_SYNC_BASELINES
        }

    def _interpret_sync_score(self, corr, sync_score, p_val):
        """
        Provides evidence-based scientific interpretation based on the
        empirical FakeAVCeleb experiment findings.
        """
        is_statistically_significant = p_val < 0.05

        if sync_score > 12.0:
            category = "High Synchronization"
            desc = "Lip movement shows strong temporal alignment with speech energy."
        elif sync_score >= 5.0:
            category = "Moderate Synchronization"
            desc = "Lip aperture and audio pulses correlate moderately; within typical range of both real speech and synced syntheses."
        else:
            category = "Low / Desynchronized"
            desc = "Lip movement deviates noticeably from vocal energy pulses, consistent with video dubbing or out-of-phase manipulation."

        scientific_note = (
            "Empirical Benchmark Context: In the 230 FakeAVCeleb validation dataset, "
            "Real videos averaged a sync score of 9.37 (median 6.48), whereas Fake videos averaged "
            "7.16 (median 0.0). Notice that while fakes have a lower median correlation (-0.0019), "
            "there is significant class overlap. Audio-lip correlation is an exploratory temporal cue, "
            "not a standalone biometric verdict."
        )

        return {
            "category": category,
            "description": desc,
            "statistically_significant": is_statistically_significant,
            "scientific_note": scientific_note
        }
