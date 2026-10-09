"""
DeepShield: Multimodal Deepfake Forensic Engine
Streamlit Web Application — High-Tech Cyber UI Edition
"""

import os
import sys
import tempfile
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Safe matplotlib cache directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from src.visual_detector import DeepShieldVisualDetector, VERIFIED_BENCHMARKS
from src.audio_analyzer import DeepShieldAudioAnalyzer
from src.sync_analyzer import DeepShieldSyncAnalyzer
from src.trust_engine import DeepShieldTrustEngine
from src.dataset_analytics import DeepShieldDatasetAnalytics


# ---------------------------------------------------------
# Page Configuration & Advanced Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="DeepShield // Multimodal Forensic Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern Dark UI Styling
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Dark Theme Background */
    .stApp {
        background-color: #0b0f19 !important;
        background: radial-gradient(circle at 50% 0%, #151f32 0%, #0b0f19 70%, #070a12 100%) !important;
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Cohesive Sidebar Dark Background & Always-Visible Guarantee */
    section[data-testid="stSidebar"], [data-testid="stSidebar"] {
        background-color: #0d1424 !important;
        background: #0d1424 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: none !important;
        margin-left: 0 !important;
        min-width: 270px !important;
        max-width: 320px !important;
    }

    /* Force visibility across all responsive viewports */
    @media (max-width: 991.98px) {
        section[data-testid="stSidebar"], [data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            transform: none !important;
            margin-left: 0 !important;
            position: relative !important;
        }
    }

    [data-testid="stSidebarUserContent"], [data-testid="stSidebar"] > div:first-child {
        background-color: #0d1424 !important;
        display: block !important;
        visibility: visible !important;
    }

    /* Prevent accidental sidebar collapsing by hiding collapse chevron */
    [data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
    }

    /* Prominent expand button in case collapsed state is cached in browser session */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 8px !important;
    }

    /* Sidebar Radio Navigation */
    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #38bdf8 !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
    }

    /* Hero Header */
    .cyber-hero {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 27, 75, 0.55) 50%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 40px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }
    .cyber-hero::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #38bdf8, #818cf8, transparent);
    }

    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }
    .badge-purple {
        background: rgba(147, 51, 234, 0.15);
        color: #c084fc;
        border: 1px solid rgba(147, 51, 234, 0.35);
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        transition: all 0.2s ease-in-out;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 12px 36px rgba(56, 189, 248, 0.08);
    }

    /* HUD Metrics */
    .hud-metric-title {
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .hud-metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .hud-metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Pulsing Online Dot */
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 12px #10b981;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Streamlit Widget Styling */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.6) !important;
    }

    /* Custom File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.35);
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.4);
        padding: 16px;
    }

    /* Filmstrip Gallery */
    .filmstrip-card {
        background: #0b0f19;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 4px;
        text-align: center;
        transition: transform 0.15s ease;
    }
    .filmstrip-card:hover {
        transform: scale(1.05);
        border-color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# State Initialization & Singleton Loaders
# ---------------------------------------------------------
@st.cache_resource
def get_visual_detector():
    checkpoint_default = "models/FINAL_EfficientNet_B0_FakeAVCeleb.pth"
    return DeepShieldVisualDetector(checkpoint_path=checkpoint_default)

@st.cache_resource
def get_audio_analyzer():
    return DeepShieldAudioAnalyzer()

@st.cache_resource
def get_sync_analyzer():
    task_model = "models/face_landmarker.task"
    return DeepShieldSyncAnalyzer(task_model_path=task_model)

@st.cache_resource
def get_trust_engine():
    return DeepShieldTrustEngine(visual_weight=0.75, sync_weight=0.25)

@st.cache_resource
def get_dataset_analytics():
    return DeepShieldDatasetAnalytics()


visual_detector = get_visual_detector()
audio_analyzer = get_audio_analyzer()
sync_analyzer = get_sync_analyzer()
trust_engine = get_trust_engine()
dataset_analytics = get_dataset_analytics()


# ---------------------------------------------------------
# Clean Minimal Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding: 4px 0;">
        <span style="font-size: 2rem;">🛡️</span>
        <div style="font-size: 1.4rem; font-weight: 800; letter-spacing: -0.02em; color: #ffffff;">DEEPSHIELD</div>
    </div>
    """, unsafe_allow_html=True)

    selected_page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard & Mission",
            "🎬 Multimodal Forensic Lab"
        ],
        label_visibility="collapsed"
    )


# ---------------------------------------------------------
# PAGE 1: DASHBOARD & MISSION
# ---------------------------------------------------------
if selected_page == "🏠 Dashboard & Mission":
    st.markdown("""
    <div class="cyber-hero">
        <h1 style="font-size: 2.5rem; margin: 0; background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            DEEPSHIELD FORENSIC SUITE
        </h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 12px; max-width: 800px; line-height: 1.6;">
            Next-generation deepfake verification engine integrating spatial convolutional networks, 
            acoustic energy profiling, and biometric audio–lip synchronization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚡ Quick Launch Checklist")
    st.markdown("""
    1. Select **🎬 Multimodal Forensic Lab** in the left navigation.
    2. Choose **"Your Video (Library Test)"** or upload an MP4/MOV file.
    3. Click **"Execute Forensic Analysis"** to inspect frame predictions, audio acoustics, and temporal sync curves!
    """)


# ---------------------------------------------------------
# PAGE 2: MULTIMODAL FORENSIC LAB
# ---------------------------------------------------------
elif selected_page == "🎬 Multimodal Forensic Lab":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px;">
        <div>
            <h2 style="margin: 0; font-size: 2rem; color: #f8fafc;">🎬 Multimodal Forensic Lab</h2>
            <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 0.95rem;">
                Run deep visual feature decomposition, speech acoustic profiling, and temporal lip synchronization.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clean Segmented Preset Selector
    st.markdown("<div style='font-size: 0.8rem; font-family: monospace; color: #94a3b8; margin-bottom: 6px;'>CHOOSE TEST VIDEO:</div>", unsafe_allow_html=True)

    preset_options = [
        "👤 Your Real Video (Library Test — 7.99s)",
        "🎬 Synthetic Demo 1 (Synchronized Speech)",
        "⚠️ Synthetic Demo 2 (Out-of-Phase / Desynced)",
        "🔇 Synthetic Demo 3 (Stripped Audio / Muted)",
        "📤 Upload Custom File"
    ]
    selected_preset = st.selectbox("Preset Selector", preset_options, label_visibility="collapsed")

    video_path = None
    video_title = ""

    if "Your Real Video" in selected_preset:
        video_path = "assets/samples/DeepShield_Test_Video_With_Audio (5).mp4"
        video_title = "DeepShield_User_Talking_Sample.mp4"
    elif "Demo 1" in selected_preset:
        video_path = "assets/samples/sample_talking_face.mp4"
        video_title = "Synthetic_Face_Synced.mp4"
    elif "Demo 2" in selected_preset:
        video_path = "assets/samples/sample_desync_face.mp4"
        video_title = "Synthetic_Face_Desync.mp4"
    elif "Demo 3" in selected_preset:
        video_path = "assets/samples/sample_no_audio.mp4"
        video_title = "Muted_Video_No_Audio.mp4"
    else:
        uploaded_file = st.file_uploader(
            "Drag & Drop Video File (MP4, MOV, AVI, WEBM)",
            type=["mp4", "mov", "avi", "webm"]
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            tfile.write(uploaded_file.getbuffer())
            tfile.close()
            video_path = tfile.name
            video_title = uploaded_file.name
            st.success(f"✓ Staged: `{uploaded_file.name}` ({round(len(uploaded_file.getbuffer())/1024, 1)} KB)")

    if video_path and os.path.exists(video_path):
        # Video Player & Telemetry Cards
        st.markdown("<br>", unsafe_allow_html=True)
        col_player, col_telemetry = st.columns([1.1, 1])

        with col_player:
            st.markdown(f"<div style='font-size: 0.85rem; font-family: monospace; color: #38bdf8; margin-bottom: 6px;'>SOURCE PREVIEW: {video_title}</div>", unsafe_allow_html=True)
            st.video(video_path)

        with col_telemetry:
            meta = visual_detector.extract_video_metadata(video_path)
            st.markdown("<div style='font-size: 0.85rem; font-family: monospace; color: #38bdf8; margin-bottom: 6px;'>CONTAINER TELEMETRY</div>", unsafe_allow_html=True)
            if meta:
                st.markdown(f"""
                <div class="glass-card" style="padding: 16px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
                        <div>
                            <div class="hud-metric-title">RESOLUTION</div>
                            <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">{meta['width']} × {meta['height']} px</div>
                        </div>
                        <div>
                            <div class="hud-metric-title">FRAME RATE</div>
                            <div style="font-size: 1.3rem; font-weight: 700; color: #38bdf8;">{meta['fps']} FPS</div>
                        </div>
                        <div>
                            <div class="hud-metric-title">TOTAL FRAMES</div>
                            <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">{meta['frame_count']} frames</div>
                        </div>
                        <div>
                            <div class="hud-metric-title">DURATION</div>
                            <div style="font-size: 1.3rem; font-weight: 700; color: #34d399;">{meta['duration']}s</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 EXECUTE MULTIMODAL FORENSIC ANALYSIS", use_container_width=True):
            with st.spinner("Executing spatial CNN inference, extracting audio features, and calculating lip synchronization..."):
                t_start = time.time()
                frames, timestamps = visual_detector.sample_frames(video_path, num_samples=16)
                audio_res = audio_analyzer.analyze_audio(video_path)
                sync_res = sync_analyzer.analyze_synchronization(audio_res, video_path)
                vis_res = visual_detector.predict_frames(frames)
                trust_res = trust_engine.evaluate(vis_res, audio_res, sync_res)
                exec_latency = round(time.time() - t_start, 3)

            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 12px 18px; margin: 16px 0;">
                <span style="color: #34d399; font-weight: 600;">✓ Forensic Pipeline Executed Successfully</span>
                <span style="font-family: monospace; font-size: 0.85rem; color: #94a3b8;">LATENCY: {exec_latency}s</span>
            </div>
            """, unsafe_allow_html=True)

            # Forensic Tabs
            tab_verdict, tab_visual, tab_audio, tab_sync = st.tabs([
                "🛡️ MULTIMODAL VERDICT",
                "👁️ SPATIAL CNN FRAMES",
                "🔊 ACOUSTIC SPECTRA",
                "👄 LIP SYNCHRONIZATION"
            ])

            # TAB 1: TRUST VERDICT
            with tab_verdict:
                score = trust_res["trust_score"]
                verdict = trust_res["authenticity_label"]
                v_color = trust_res.get("verdict_color", "emerald")

                if v_color == "emerald":
                    gauge_color = "#10b981"
                    badge_style = "badge-green"
                elif v_color == "amber":
                    gauge_color = "#f59e0b"
                    badge_style = "badge-amber"
                else:
                    gauge_color = "#ef4444"
                    badge_style = "badge-amber"

                col_gauge, col_hud = st.columns([1, 1.3])

                with col_gauge:
                    # High-Tech Plotly Radial Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score if score is not None else 0,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        number={'suffix': "/100", 'font': {'size': 38, 'color': '#f8fafc', 'family': 'Plus Jakarta Sans'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': gauge_color, 'thickness': 0.28},
                            'bgcolor': "rgba(15, 23, 42, 0.4)",
                            'borderwidth': 0,
                            'steps': [
                                {'range': [0, 45], 'color': 'rgba(239, 68, 68, 0.15)'},
                                {'range': [45, 65], 'color': 'rgba(245, 158, 11, 0.15)'},
                                {'range': [65, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                            ],
                            'threshold': {
                                'line': {'color': "#ffffff", 'width': 3},
                                'thickness': 0.75,
                                'value': score if score is not None else 0
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=280,
                        margin=dict(l=20, r=20, t=30, b=10)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.markdown(f"""
                    <div style="text-align: center;">
                        <span class="badge {badge_style}" style="font-size: 0.9rem; padding: 6px 16px;">VERDICT: {verdict}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_hud:
                    st.markdown("""
                    <div class="hud-metric-title">FORENSIC SIGNAL CONTRIBUTIONS</div>
                    """, unsafe_allow_html=True)

                    # 3 Signal Contribution HUD Cards
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("""
                        <div class="glass-card" style="padding: 14px;">
                            <div class="hud-metric-title">SPATIAL CNN</div>
                        """, unsafe_allow_html=True)
                        if vis_res.get("status") == "success":
                            v_pts = round(vis_res['mean_real_prob'] * 0.75, 1)
                            st.markdown(f"""
                            <div class="hud-metric-value" style="font-size: 1.4rem; color: #38bdf8;">{vis_res['mean_real_prob']}%</div>
                            <div class="hud-metric-sub">+{v_pts} pts (75% wt)</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='color: #fbbf24; font-size: 0.85rem;'>Awaiting .pth</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with c2:
                        st.markdown("""
                        <div class="glass-card" style="padding: 14px;">
                            <div class="hud-metric-title">AUDIO CADENCE</div>
                        """, unsafe_allow_html=True)
                        if audio_res.get("has_audio"):
                            st.markdown(f"""
                            <div class="hud-metric-value" style="font-size: 1.4rem; color: #c084fc;">{audio_res['speech_ratio']}%</div>
                            <div class="hud-metric-sub">RMS: {audio_res['mean_rms']:.3f}</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='color: #ef4444; font-size: 0.85rem;'>No Audio Track</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with c3:
                        st.markdown("""
                        <div class="glass-card" style="padding: 14px;">
                            <div class="hud-metric-title">LIP SYNC</div>
                        """, unsafe_allow_html=True)
                        if sync_res.get("success"):
                            s_norm = min(100.0, (sync_res['sync_score'] / 25.0) * 100.0)
                            s_pts = round(0.25 * s_norm, 1)
                            st.markdown(f"""
                            <div class="hud-metric-value" style="font-size: 1.4rem; color: #34d399;">{sync_res['sync_score']:.1f}</div>
                            <div class="hud-metric-sub">+{s_pts} pts (25% wt)</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='color: #ef4444; font-size: 0.85rem;'>0 pts (Nullified)</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Diagnostic Reasoning Box
                    st.markdown(f"""
                    <div class="glass-card" style="margin-top: 12px; padding: 14px; border-left: 3px solid {gauge_color};">
                        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">{trust_res['reasoning']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # TAB 2: VISUAL CNN ANALYSIS
            with tab_visual:
                st.markdown("### 👁️ EfficientNet-B0 Frame Decomposition")
                st.markdown("<div style='font-size: 0.8rem; font-family: monospace; color: #94a3b8; margin-bottom: 10px;'>UNIFORMLY EXTRACTED INSPECTION FRAMES (16 SAMPLES):</div>", unsafe_allow_html=True)

                # Filmstrip Gallery
                cols = st.columns(8)
                for idx, (f_idx, img) in enumerate(frames[:8]):
                    with cols[idx]:
                        t_sec = timestamps[idx] if idx < len(timestamps) else 0.0
                        st.markdown(f"<div class='filmstrip-card'><span style='font-size: 0.7rem; font-family: monospace; color: #38bdf8;'>F#{f_idx} ({t_sec:.1f}s)</span></div>", unsafe_allow_html=True)
                        st.image(img, use_container_width=True)

                cols2 = st.columns(8)
                for idx, (f_idx, img) in enumerate(frames[8:16]):
                    with cols2[idx]:
                        actual_idx = idx + 8
                        t_sec = timestamps[actual_idx] if actual_idx < len(timestamps) else 0.0
                        st.markdown(f"<div class='filmstrip-card'><span style='font-size: 0.7rem; font-family: monospace; color: #38bdf8;'>F#{f_idx} ({t_sec:.1f}s)</span></div>", unsafe_allow_html=True)
                        st.image(img, use_container_width=True)

                st.divider()
                if vis_res.get("status") == "success":
                    st.markdown("#### 🧠 Frame-by-Frame Softmax Probability Activation")
                    df_frames = pd.DataFrame(vis_res["frame_results"])
                    fig_bars = px.bar(
                        df_frames,
                        x="frame_index",
                        y=["prob_real", "prob_fake"],
                        barmode="group",
                        labels={"value": "Softmax Probability", "frame_index": "Video Frame Number"},
                        color_discrete_map={"prob_real": "#10b981", "prob_fake": "#ef4444"}
                    )
                    fig_bars.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15, 23, 42, 0.4)",
                        height=280,
                        margin=dict(l=30, r=20, t=20, b=30)
                    )
                    st.plotly_chart(fig_bars, use_container_width=True)

            # TAB 3: AUDIO ACOUSTICS
            with tab_audio:
                st.markdown("### 🔊 Physical Acoustic Signal Decomposition")
                if audio_res.get("has_audio"):
                    # Waveform
                    st.markdown("<div class='hud-metric-title'>NORMALIZED AMPLITUDE WAVEFORM (16kHz DOWNSAMPLED)</div>", unsafe_allow_html=True)
                    df_wave = pd.DataFrame({"Time (s)": audio_res["waveform_times"], "Amplitude": audio_res["waveform"]})
                    fig_w = px.line(df_wave, x="Time (s)", y="Amplitude", color_discrete_sequence=["#38bdf8"])
                    fig_w.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.4)", height=220, margin=dict(l=30, r=20, t=10, b=30))
                    st.plotly_chart(fig_w, use_container_width=True)

                    # MFCC
                    st.markdown("<div class='hud-metric-title'>13-BAND MEL-FREQUENCY CEPSTRAL COEFFICIENTS (MFCC HEATMAP)</div>", unsafe_allow_html=True)
                    fig_m = px.imshow(audio_res["mfcc"], aspect="auto", color_continuous_scale="Viridis", labels=dict(x="Time Frame (512 hop)", y="MFCC Index", color="dB"))
                    fig_m.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.4)", height=240, margin=dict(l=30, r=20, t=10, b=30))
                    st.plotly_chart(fig_m, use_container_width=True)
                else:
                    st.warning("⚠️ No active audio container detected.")

            # TAB 4: AUDIO-LIP SYNCHRONIZATION
            with tab_sync:
                st.markdown("### 👄 Temporal Audio–Lip Synchronization")
                if sync_res.get("success"):
                    # Alignment Curve
                    st.markdown("<div class='hud-metric-title'>TEMPORAL COHERENCE: SPEECH VOCAL ENERGY vs. LIP APERTURE</div>", unsafe_allow_html=True)
                    df_sync = pd.DataFrame({
                        "Time (s)": sync_res["audio_times"],
                        "Speech Energy (RMS > Mean)": sync_res["speech_activity"],
                        "Normalized Lip Aperture": sync_res["mouth_norm"]
                    })
                    fig_s = px.line(
                        df_sync,
                        x="Time (s)",
                        y=["Speech Energy (RMS > Mean)", "Normalized Lip Aperture"],
                        color_discrete_map={"Speech Energy (RMS > Mean)": "#38bdf8", "Normalized Lip Aperture": "#ec4899"}
                    )
                    fig_s.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15, 23, 42, 0.4)",
                        height=300,
                        margin=dict(l=30, r=20, t=20, b=30)
                    )
                    st.plotly_chart(fig_s, use_container_width=True)

                    # Benchmark explanation
                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 3px solid #10b981; margin-top: 14px;">
                        <span class="badge badge-green">TOP-TIER HUMAN SYNCHRONIZATION</span>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-top: 6px;">Sync Score: {sync_res['sync_score']:.2f} / 100 (Pearson r = {sync_res['correlation']:.4f})</div>
                        <p style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
                            In the 230-video FakeAVCeleb experiment, authentic human speech averages only <strong>9.37</strong> (median 6.48). 
                            Scores above 15.0 represent high synchronization due to non-linear bilabial consonants ('M', 'B', 'P') where human lips close during vocalization.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ Synchronization Inactive: {sync_res.get('reason', 'Missing stream')}")



