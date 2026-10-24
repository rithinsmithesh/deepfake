"""
Live Video & Audio Recorder Component Wrapper for DeepShield
Bridges the browser-based HTML5 MediaRecorder component with Streamlit Python backend.
Standardizes live camera & microphone video feeds into MP4/AAC for multimodal forensic evaluation.
"""

import os
import base64
import tempfile
import subprocess
import imageio_ffmpeg
import streamlit as st
import streamlit.components.v1 as components


_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webcam_recorder")

if os.path.exists(_COMPONENT_DIR):
    _webcam_recorder_component = components.declare_component(
        "deepshield_webcam_recorder",
        path=_COMPONENT_DIR
    )
else:
    _webcam_recorder_component = None


def render_live_webcam_recorder(key="deepshield_live_recorder"):
    """
    Renders the browser-based HTML5 camera & microphone recorder.
    Returns:
        dict with keys:
            'video_path': path to converted MP4 file (or None if no recording yet)
            'duration': duration in seconds
            'raw_size': size in bytes
            'timestamp': recording timestamp
    """
    if _webcam_recorder_component is None:
        st.error("Live recorder component directory not found.")
        return None

    result = _webcam_recorder_component(key=key)

    if not result or not isinstance(result, dict) or "base64" not in result:
        return None

    b64_str = result.get("base64", "")
    if not b64_str:
        return None

    # Strip Data URL header if present (e.g. "data:video/webm;base64,...")
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(b64_str)
    except Exception as e:
        st.error(f"Failed to decode live recorded video stream: {e}")
        return None

    if len(raw_bytes) < 1000:
        return None

    # Save raw video to temp file
    raw_suffix = ".webm" if "webm" in result.get("mime_type", "webm") else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=raw_suffix) as temp_raw:
        temp_raw.write(raw_bytes)
        raw_path = temp_raw.name

    # Standardize to MP4 (H.264 / AAC) using imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_mp4_path = out_mp4.name
    out_mp4.close()

    convert_cmd = [
        ffmpeg_exe,
        "-y",
        "-i", raw_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-c:a", "aac",
        "-ar", "16000",
        "-ac", "1",
        "-movflags", "+faststart",
        out_mp4_path
    ]

    try:
        conv_res = subprocess.run(
            convert_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if conv_res.returncode == 0 and os.path.exists(out_mp4_path) and os.path.getsize(out_mp4_path) > 1000:
            final_path = out_mp4_path
        else:
            final_path = raw_path
    except Exception:
        final_path = raw_path

    return {
        "video_path": final_path,
        "duration": result.get("duration_sec", 0.0),
        "raw_size": len(raw_bytes),
        "mime_type": result.get("mime_type", "video/webm"),
        "timestamp": result.get("timestamp")
    }
