"""Live browser-camera panel for DeepShield."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer


def _rtc_configuration() -> dict[str, Any]:
    """Build browser/server ICE configuration without exposing credentials."""
    ice_servers: list[dict[str, Any]] = [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]

    # TURN is required on some Streamlit Community Cloud, office, and mobile
    # networks. Configure these three values in Streamlit Cloud Secrets.
    try:
        turn_urls = st.secrets.get("TURN_URLS")
        turn_username = st.secrets.get("TURN_USERNAME")
        turn_credential = st.secrets.get("TURN_CREDENTIAL")
    except Exception:
        turn_urls = turn_username = turn_credential = None

    if turn_urls and turn_username and turn_credential:
        if isinstance(turn_urls, str):
            turn_urls = [item.strip() for item in turn_urls.split(",") if item.strip()]
        ice_servers.append(
            {
                "urls": list(turn_urls),
                "username": str(turn_username),
                "credential": str(turn_credential),
            }
        )

    return {"iceServers": ice_servers}


class LiveCameraProcessor(VideoProcessorBase):
    """Echo the camera stream and retain a rolling frame buffer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: deque[np.ndarray] = deque(maxlen=24)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._frames.append(image.copy())
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def snapshot(self, count: int = 8) -> list[tuple[int, np.ndarray]]:
        with self._lock:
            frames = list(self._frames)[-count:]
        if not frames:
            return []
        indexes = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
        return [(int(index), frames[index]) for index in indexes]


def _visual_live_result(visual_detector: Any, frames: list[tuple[int, np.ndarray]]) -> dict[str, Any]:
    if not frames:
        return {"status": "error", "reason": "No camera frames received yet."}
    try:
        result = visual_detector.predict_frames(frames)
        result["live_mode"] = True
        result["frames_analyzed"] = len(frames)
        return result
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "live_mode": True}


def render_live_camera(visual_detector: Any) -> None:
    """Render the permission-gated Start / Execute / Stop workflow."""
    st.markdown("### 📷 Live Camera Forensic Capture")
    st.caption(
        "Start the browser camera to preview live frames. Execute analyzes recent visual frames; "
        "Stop releases the camera and ends the stream."
    )
    st.info(
        "Press Start and allow camera access when your browser asks. On a phone, use Chrome or Safari "
        "over HTTPS and allow camera access for deepfakeaudio.streamlit.app."
    )

    context = webrtc_streamer(
        key="deepshield-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveCameraProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "user"},
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
            },
            "audio": False,
        },
        async_processing=True,
        rtc_configuration=_rtc_configuration(),
    )

    processor = context.video_processor
    is_running = bool(context.state.playing and processor is not None)
    if not is_running:
        st.caption(
            "If the stream stays disconnected: check the browser camera permission, tap Select Device "
            "to choose a camera, then retry on another Wi-Fi or mobile network. TURN credentials are "
            "needed for some restricted networks."
        )

    execute = st.button(
        "🚀 EXECUTE LIVE FRAME ANALYSIS",
        type="primary",
        use_container_width=True,
        disabled=not is_running,
        help="Start the camera first so recent frames are available.",
    )

    if execute and processor is not None:
        st.session_state["deepshield_live_result"] = _visual_live_result(
            visual_detector, processor.snapshot()
        )

    result = st.session_state.get("deepshield_live_result")
    if result:
        st.divider()
        if result.get("status") == "success":
            st.success("Live visual analysis completed.")
            c1, c2 = st.columns(2)
            c1.metric("Frames analyzed", result.get("frames_analyzed", 0))
            c2.metric("Mean real probability", f"{result.get('mean_real_prob', 0):.2f}%")
            st.json(result)
        else:
            st.warning(f"Live analysis unavailable: {result.get('reason', 'Unknown error')}")
