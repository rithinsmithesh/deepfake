"""Live camera and video-call capture panel for DeepShield."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import av
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer


def _rtc_configuration() -> dict[str, Any]:
    """Build browser/server ICE configuration without exposing credentials."""
    ice_servers: list[dict[str, Any]] = [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
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
    """Echo camera frames and retain a rolling buffer for forensic capture."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: deque[np.ndarray] = deque(maxlen=48)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._frames.append(image.copy())
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def snapshot(self, count: int = 16) -> list[tuple[int, np.ndarray]]:
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


def _capture_live_window(processor: LiveCameraProcessor, seconds: int) -> list[tuple[int, np.ndarray]]:
    """Keep the live stream running while the user records a short call window."""
    timer = st.empty()
    started = time.time()
    while time.time() - started < seconds:
        elapsed = min(seconds, time.time() - started)
        timer.markdown(
            f"<div class='live-call-timer'>00:{elapsed:04.1f} / 00:{seconds:02d}.0</div>",
            unsafe_allow_html=True,
        )
        time.sleep(0.25)
    timer.empty()
    return processor.snapshot(count=16)


def _render_result(result: dict[str, Any]) -> None:
    st.divider()
    if result.get("status") != "success":
        st.warning(f"Live analysis unavailable: {result.get('reason', 'Unknown error')}")
        return

    st.success("✓ Forensic Analysis Executed")
    frame_results = result.get("frame_results", [])
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Frames analyzed", result.get("frames_analyzed", len(frame_results)))
    metric_b.metric("Real probability", f"{result.get('mean_real_prob', 0):.2f}%")
    metric_c.metric("Capture mode", "Live camera")

    if frame_results:
        frame_df = pd.DataFrame(frame_results)
        if {"frame_index", "prob_real", "prob_fake"}.issubset(frame_df.columns):
            st.markdown("#### Live frame probability timeline")
            st.line_chart(frame_df.set_index("frame_index")[["prob_real", "prob_fake"]])
    with st.expander("View forensic frame details"):
        st.json(result)


def render_live_camera(visual_detector: Any, capture_mode: str = "video_call") -> None:
    """Render a permission-gated live video-call or instant-snapshot workflow."""
    is_snapshot_mode = capture_mode == "snapshot"
    st.markdown(
        """
        <style>
        .live-call-shell { background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(15,32,61,.92)); border: 1px solid rgba(56,189,248,.42); border-radius: 18px; padding: 22px 24px 10px; box-shadow: 0 16px 50px rgba(0,0,0,.35); }
        .live-call-title { color:#38bdf8; font-family:monospace; font-size:1.25rem; font-weight:800; letter-spacing:.06em; }
        .live-call-help { color:#94a3b8; margin-top:8px; line-height:1.5; }
        .live-call-status { display:inline-flex; align-items:center; gap:10px; color:#34d399; border:1px solid rgba(16,185,129,.55); border-radius:999px; padding:8px 16px; font-weight:700; margin:12px 0; }
        .live-call-status.waiting { color:#fbbf24; border-color:rgba(245,158,11,.55); }
        .live-call-dot { width:11px; height:11px; border-radius:50%; background:#34d399; box-shadow:0 0 12px #34d399; }
        .live-call-status.waiting .live-call-dot { background:#fbbf24; box-shadow:0 0 12px #fbbf24; }
        .live-call-timer { color:#38bdf8; font-family:monospace; font-size:1.7rem; font-weight:800; text-align:right; margin:8px 0; }
        @media (max-width: 768px) { .live-call-shell { padding:16px 14px 8px; } .live-call-title { font-size:1rem; } .live-call-timer { text-align:left; font-size:1.3rem; } }
        </style>
        <div class="live-call-shell">
            <div class="live-call-title">🛡️ LIVE VIDEO CALL RECORDER</div>
            <div class="live-call-help">Allow camera and microphone access, confirm the live preview, then capture a short forensic window.</div>
        </div>
        """,
        unsafe_allow_html=True,
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
            "audio": True,
        },
        async_processing=True,
        rtc_configuration=_rtc_configuration(),
    )

    processor = context.video_processor
    is_running = bool(context.state.playing and processor is not None)
    status_class = "" if is_running else "waiting"
    status_text = "LIVE CAMERA READY" if is_running else "WAITING FOR CAMERA & MICROPHONE"
    st.markdown(
        f"<div class='live-call-status {status_class}'><span class='live-call-dot'></span>{status_text}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Use the component Start button to request permission. Its Stop button releases the camera and microphone."
    )

    if not is_running:
        st.info(
            "On first use, choose Allow while visiting the site. If the stream does not connect, use Select Device, "
            "check browser permissions, and retry on another network."
        )

    duration_options = {
        "8 Seconds (Recommended)": 8,
        "15 Seconds": 15,
        "30 Seconds": 30,
    }
    duration_label = st.selectbox(
        "Capture duration",
        list(duration_options),
        index=0,
        disabled=not is_running,
    )
    duration = duration_options[duration_label]

    if is_snapshot_mode:
        capture = st.button(
            "📸 CAPTURE INSTANT SNAPSHOT",
            type="primary",
            use_container_width=True,
            disabled=not is_running,
        )
        if capture and processor is not None:
            st.session_state["deepshield_live_result"] = _visual_live_result(
                visual_detector, processor.snapshot(count=8)
            )
    else:
        capture = st.button(
            "🔴 START RECORDING CALL",
            type="primary",
            use_container_width=True,
            disabled=not is_running,
        )
        if capture and processor is not None:
            st.session_state["deepshield_live_result"] = _visual_live_result(
                visual_detector, _capture_live_window(processor, duration)
            )

    result = st.session_state.get("deepshield_live_result")
    if result:
        _render_result(result)
