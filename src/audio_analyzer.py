"""
Audio Analysis Module for DeepShield
Extracts audio from video containers and computes physical acoustic features:
Waveform, RMS Speech Activity Envelope, Mel-Frequency Cepstral Coefficients (MFCCs),
and Spectral Characteristics using librosa and ffmpeg.
"""

import os
import subprocess
import tempfile
import numpy as np
import librosa
import imageio_ffmpeg


class DeepShieldAudioAnalyzer:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def extract_audio(self, video_path):
        """
        Extracts audio track from video file into 16kHz mono WAV.
        Returns temporary file path or None if no audio track exists.
        """
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()

        command = [
            self.ffmpeg_exe,
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(self.target_sr),
            temp_wav_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode != 0 or not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) < 200:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            return None

        return temp_wav_path

    def analyze_audio(self, video_path):
        """
        Performs genuine physical audio feature extraction on video audio track.
        """
        wav_path = self.extract_audio(video_path)

        if not wav_path:
            return {
                "has_audio": False,
                "message": "No audio stream detected in video container.",
                "features": None
            }

        try:
            # Load audio using librosa
            y, sr = librosa.load(wav_path, sr=self.target_sr, mono=True)
            duration = float(len(y) / sr)

            if len(y) < sr * 0.2:  # Less than 200ms
                return {
                    "has_audio": False,
                    "message": "Audio duration is too short for meaningful acoustic feature analysis (<0.2s).",
                    "features": None
                }

            # 1. RMS Energy & Speech Activity (exact formula from deepshield-member3 notebook)
            hop_length = 512
            frame_length = 2048
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            rms_threshold = float(np.mean(rms))
            speech_activity = (rms > rms_threshold).astype(float)
            audio_times = librosa.frames_to_time(np.arange(len(speech_activity)), sr=sr, hop_length=hop_length)

            # 2. MFCCs (13 coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

            # 3. Spectral Centroid and Zero Crossing Rate
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]

            # 4. Downsampled waveform for responsive UI rendering
            downsample_factor = max(1, len(y) // 2000)
            waveform_sub = y[::downsample_factor]
            waveform_times = np.linspace(0, duration, len(waveform_sub))

            return {
                "has_audio": True,
                "duration": round(duration, 2),
                "sample_rate": sr,
                "total_samples": len(y),
                "mean_rms": float(rms_threshold),
                "speech_ratio": round(float(np.mean(speech_activity) * 100), 2),
                "audio_times": audio_times,
                "speech_activity": speech_activity,
                "rms_envelope": rms,
                "mfcc": mfcc,
                "spectral_centroid": spectral_centroid,
                "zcr": zcr,
                "waveform": waveform_sub,
                "waveform_times": waveform_times,
                "wav_file_path": wav_path,
                "disclaimer": (
                    "Acoustic Feature Extraction: Physical signal characteristics (MFCCs, energy envelope, spectral centroid) "
                    "measure voice presence and frequency dynamics. This module does not claim to be a standalone AI synthetic "
                    "voice discriminator without a dedicated acoustic deepfake neural network."
                )
            }
        except Exception as e:
            return {
                "has_audio": False,
                "message": f"Audio processing error: {str(e)}",
                "features": None
            }
