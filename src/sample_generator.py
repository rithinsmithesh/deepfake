"""
Sample Video Generator for DeepShield Demo
Generates synthetic talking-face test videos with audio for immediate local demonstration.
"""

import os
import cv2
import numpy as np
import soundfile as sf
import subprocess
import imageio_ffmpeg


def generate_sample_videos(output_dir="assets/samples"):
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    fps = 25
    duration = 4.0  # 4 seconds
    total_frames = int(fps * duration)
    width, height = 320, 320
    sample_rate = 16000

    samples = [
        {"name": "sample_talking_face.mp4", "synced": True, "has_audio": True},
        {"name": "sample_desync_face.mp4", "synced": False, "has_audio": True},
        {"name": "sample_no_audio.mp4", "synced": True, "has_audio": False},
    ]

    for sample in samples:
        target_path = os.path.join(output_dir, sample["name"])
        if os.path.exists(target_path):
            continue

        temp_video = os.path.join(output_dir, f"_temp_{sample['name']}")
        temp_audio = os.path.join(output_dir, f"_temp_{sample['name']}.wav")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        # Audio time array
        t_audio = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Pulse frequency: 1.5 Hz talking rhythm
        speech_envelope = np.maximum(0, np.sin(2 * np.pi * 1.5 * t_audio))
        audio_carrier = np.sin(2 * np.pi * 220 * t_audio) * 0.4
        audio_signal = (speech_envelope * audio_carrier).astype(np.float32)

        if sample["has_audio"]:
            sf.write(temp_audio, audio_signal, sample_rate)

        # Video frames
        for f in range(total_frames):
            frame_time = f / fps
            img = np.full((height, width, 3), (35, 25, 25), dtype=np.uint8)

            # Draw a face
            center_x, center_y = width // 2, height // 2 - 10
            # Head contour
            cv2.ellipse(img, (center_x, center_y), (85, 110), 0, 0, 360, (210, 185, 170), -1)
            cv2.ellipse(img, (center_x, center_y), (85, 110), 0, 0, 360, (140, 110, 95), 2)

            # Eyes
            cv2.circle(img, (center_x - 32, center_y - 25), 8, (60, 40, 30), -1)
            cv2.circle(img, (center_x + 32, center_y - 25), 8, (60, 40, 30), -1)

            # Nose
            cv2.line(img, (center_x, center_y - 10), (center_x, center_y + 15), (150, 120, 100), 2)

            # Mouth
            mouth_y = center_y + 45
            if sample["synced"]:
                # Mouth opening in sync with 1.5 Hz speech envelope
                mouth_open = max(2, int(20 * max(0, np.sin(2 * np.pi * 1.5 * frame_time))))
            else:
                # Out of phase (desynchronized by 90 degrees or different frequency)
                mouth_open = max(2, int(20 * max(0, np.cos(2 * np.pi * 1.5 * frame_time + np.pi/2))))

            # Upper and lower lips
            cv2.ellipse(img, (center_x, mouth_y), (25, mouth_open), 0, 0, 360, (50, 30, 120), -1)
            cv2.ellipse(img, (center_x, mouth_y), (25, mouth_open), 0, 0, 360, (30, 15, 80), 2)

            out.write(img)

        out.release()

        # Combine video and audio using ffmpeg
        if sample["has_audio"]:
            cmd = [
                ffmpeg_exe, "-y",
                "-i", temp_video,
                "-i", temp_audio,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-shortest",
                target_path
            ]
        else:
            cmd = [
                ffmpeg_exe, "-y",
                "-i", temp_video,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                target_path
            ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(temp_video):
            os.remove(temp_video)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    print("Demo sample videos ready in:", output_dir)


if __name__ == "__main__":
    generate_sample_videos()
