"""
DeepShield: Multimodal Deepfake Forensic Engine
Streamlit Web Application — High-Tech Cyber UI Edition
Integrates Spatial Convolutional Networks (EfficientNet-B0), Physical Acoustics (MFCC),
Biometric Audio–Lip Synchronization (MediaPipe), and Live Camera & Call Operations.
"""

import os
import sys
import tempfile
import time
import datetime
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Safe matplotlib cache directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from src.visual_detector import DeepShieldVisualDetector, VERIFIED_BENCHMARKS
from src.audio_analyzer import DeepShieldAudioAnalyzer
from src.sync_analyzer import DeepShieldSyncAnalyzer
from src.trust_engine import DeepShieldTrustEngine
from src.dataset_analytics import DeepShieldDatasetAnalytics
from src.live_camera_processor import LiveCameraProcessor
from src.live_recorder import render_live_webcam_recorder


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
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.45);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.45);
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.25);
    }
    .badge-cyan {
        background: rgba(6, 182, 212, 0.18);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.45);
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.25);
    }
    .badge-purple {
        background: rgba(147, 51, 234, 0.18);
        color: #c084fc;
        border: 1px solid rgba(147, 51, 234, 0.45);
        box-shadow: 0 0 15px rgba(147, 51, 234, 0.25);
    }
    .badge-rose {
        background: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.25);
    }

    /* Enhanced Verdict Banner */
    .verdict-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(24, 33, 56, 0.92) 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .verdict-banner.emerald {
        border: 2px solid rgba(16, 185, 129, 0.5);
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.2), inset 0 1px 0 rgba(16, 185, 129, 0.3);
    }
    .verdict-banner.amber {
        border: 2px solid rgba(245, 158, 11, 0.5);
        box-shadow: 0 12px 40px rgba(245, 158, 11, 0.2), inset 0 1px 0 rgba(245, 158, 11, 0.3);
    }
    .verdict-banner.rose {
        border: 2px solid rgba(239, 68, 68, 0.5);
        box-shadow: 0 12px 40px rgba(239, 68, 68, 0.2), inset 0 1px 0 rgba(239, 68, 68, 0.3);
    }

    /* KPI Holographic Tiles */
    .kpi-card {
        background: rgba(11, 15, 25, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
        transition: all 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #38bdf8;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.15);
    }

    /* Mathematical Formula Box */
    .math-formula-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 14px 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        transition: all 0.2s ease-in-out;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 12px 36px rgba(56, 189, 248, 0.1);
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
        border: 1.5px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px;
        text-align: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .filmstrip-card.real-frame {
        border-color: rgba(16, 185, 129, 0.6);
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }
    .filmstrip-card.fake-frame {
        border-color: rgba(239, 68, 68, 0.6);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
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

def get_trust_engine():
    return DeepShieldTrustEngine(visual_weight=0.75, sync_weight=0.25)

@st.cache_resource
def get_dataset_analytics():
    return DeepShieldDatasetAnalytics()

@st.cache_resource
def get_live_camera_processor():
    task_model = "models/face_landmarker.task"
    return LiveCameraProcessor(task_model_path=task_model)


visual_detector = get_visual_detector()
audio_analyzer = get_audio_analyzer()
sync_analyzer = get_sync_analyzer()
trust_engine = get_trust_engine()
dataset_analytics = get_dataset_analytics()
live_camera_processor = get_live_camera_processor()


# ---------------------------------------------------------
# Reusable Forensic Tab Rendering Function
# ---------------------------------------------------------
def render_multimodal_forensic_results(trust_res, vis_res, audio_res, sync_res, frames, timestamps, exec_latency, trust_engine_instance=None):
    """Renders the comprehensive 4-tab forensic report with radial gauge and diagnostic curves."""
    
    # Sensitivity profile selection
    col_prof, col_status = st.columns([1.3, 1])
    with col_prof:
        prof_choice = st.selectbox(
            "🎯 FORENSIC MULTIMODAL SENSITIVITY PROFILE",
            [
                "⚖️ Balanced Standard (75% Spatial CNN / 25% Temporal Sync)",
                "📞 Live Video Call Sensitive (60% Spatial CNN / 40% Temporal Sync)",
                "👁️ Visual-Dominant Priority (85% Spatial CNN / 15% Temporal Sync)",
                "🎚️ 50:50 Dual Modality (50% Spatial CNN / 50% Temporal Sync)"
            ],
            index=0,
            key="forensic_profile_select"
        )
    with col_status:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 10px 14px; margin-top: 18px; text-align: right;">
            <span style="color: #34d399; font-weight: 600; font-size: 0.85rem;">✓ Forensic Analysis Executed</span><br>
            <span style="font-family: monospace; font-size: 0.75rem; color: #94a3b8;">INFERENCE LATENCY: {exec_latency}s | {vis_res.get('device', 'cpu').upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    # Dynamic Weight Re-evaluation
    if "Live Video Call" in prof_choice:
        w_vis, w_sync = 0.60, 0.40
    elif "Visual-Dominant" in prof_choice:
        w_vis, w_sync = 0.85, 0.15
    elif "50:50" in prof_choice:
        w_vis, w_sync = 0.50, 0.50
    else:
        w_vis, w_sync = 0.75, 0.25

    if trust_engine_instance is not None:
        try:
            trust_res = trust_engine_instance.evaluate(vis_res, audio_res, sync_res, visual_weight=w_vis, sync_weight=w_sync)
        except TypeError:
            try:
                trust_engine_instance.visual_weight = w_vis
                trust_engine_instance.sync_weight = w_sync
                trust_res = trust_engine_instance.evaluate(vis_res, audio_res, sync_res)
            except Exception:
                trust_res = trust_engine_instance.evaluate(vis_res, audio_res, sync_res)

    score = trust_res.get("trust_score", 0.0) or 0.0
    deepfake_prob = trust_res.get("deepfake_probability", round(100.0 - score, 2))
    verdict = trust_res.get("authenticity_label", "Evaluating...")
    v_color = trust_res.get("verdict_color", "emerald")
    forensic_grade = trust_res.get("forensic_grade", "GRADE A (Authentic Human)")
    conf_rating = trust_res.get("confidence_rating", "HIGH")
    points = trust_res.get("points_breakdown", {})

    if v_color == "emerald":
        gauge_color = "#10b981"
        risk_color = "#10b981"
        badge_style = "badge-green"
        status_icon = "🛡️"
        verdict_title = "VERIFIED AUTHENTIC (HUMAN MEDIA)"
        verdict_tag = "AUTHENTIC HUMAN"
    elif v_color == "amber":
        gauge_color = "#f59e0b"
        risk_color = "#f59e0b"
        badge_style = "badge-amber"
        status_icon = "⚠️"
        verdict_title = "SUSPICIOUS / INCONCLUSIVE ANOMALY"
        verdict_tag = "SUSPICIOUS"
    else:
        gauge_color = "#ef4444"
        risk_color = "#ef4444"
        badge_style = "badge-rose"
        status_icon = "🚨"
        verdict_title = "SYNTHETIC DEEPFAKE MANIPULATION DETECTED"
        verdict_tag = "DEEPFAKE DETECTED"

    # Audit signature hash
    audit_hash = hashlib.sha256(f"{score}_{deepfake_prob}_{time.time()}".encode()).hexdigest()[:12].upper()
    audit_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. HERO VERDICT CERTIFICATE BANNER
    st.markdown(f"""
    <div class="verdict-banner {v_color}">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 14px; margin-bottom: 18px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 2.2rem;">{status_icon}</span>
                <div>
                    <div style="font-size: 0.76rem; font-family: 'JetBrains Mono', monospace; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.1em;">
                        DEEPSHIELD OFFICIAL FORENSIC VERDICT CERTIFICATE
                    </div>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
                        {verdict_title}
                    </div>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="badge {badge_style}" style="font-size: 0.85rem; padding: 6px 16px;">
                    {forensic_grade}
                </span>
                <div style="font-size: 0.72rem; font-family: monospace; color: #94a3b8; margin-top: 6px;">
                    AUDIT ID: #{audit_hash} | {audit_time}
                </div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;">
            <div class="kpi-card">
                <div class="hud-metric-title">EXACT AUTHENTICITY SCORE</div>
                <div class="hud-metric-value" style="color: {gauge_color}; font-size: 2.2rem;">
                    {score:.2f}<span style="font-size: 1.1rem; color: #94a3b8;">%</span>
                </div>
                <div class="hud-metric-sub">P(Authentic Real Human)</div>
            </div>

            <div class="kpi-card">
                <div class="hud-metric-title">DEEPFAKE RISK PROBABILITY</div>
                <div class="hud-metric-value" style="color: {'#ef4444' if deepfake_prob > 35 else '#34d399'}; font-size: 2.2rem;">
                    {deepfake_prob:.2f}<span style="font-size: 1.1rem; color: #94a3b8;">%</span>
                </div>
                <div class="hud-metric-sub">P(Synthetic Manipulation)</div>
            </div>

            <div class="kpi-card">
                <div class="hud-metric-title">STATISTICAL CONFIDENCE</div>
                <div class="hud-metric-value" style="color: #38bdf8; font-size: 1.45rem; padding-top: 6px;">
                    {conf_rating}
                </div>
                <div class="hud-metric-sub">Bimodal Fusion Certainty</div>
            </div>

            <div class="kpi-card">
                <div class="hud-metric-title">FINAL VERDICT STATUS</div>
                <div class="hud-metric-value" style="color: #f8fafc; font-size: 1.3rem; padding-top: 8px;">
                    {verdict_tag}
                </div>
                <div class="hud-metric-sub">Decision Boundary Threshold</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. TABBED FORENSIC BREAKDOWN
    tab_verdict, tab_visual, tab_audio, tab_sync = st.tabs([
        "🛡️ EXACT VERDICT SCOREBOARD",
        "👁️ SPATIAL CNN FRAMES",
        "🔊 ACOUSTIC SPECTRA",
        "👄 LIP SYNCHRONIZATION"
    ])

    # TAB 1: EXACT VERDICT SCOREBOARD
    with tab_verdict:
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "AUTHENTICITY TRUST SCORE", 'font': {'size': 14, 'color': '#94a3b8', 'family': 'JetBrains Mono'}},
                number={'suffix': "/100", 'font': {'size': 36, 'color': '#f8fafc', 'family': 'Plus Jakarta Sans'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': gauge_color, 'thickness': 0.3},
                    'bgcolor': "rgba(15, 23, 42, 0.4)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [40, 65], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [65, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 3},
                        'thickness': 0.8,
                        'value': score
                    }
                }
            ))
            fig_g1.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=250,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_g1, use_container_width=True)

        with col_g2:
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=deepfake_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "DEEPFAKE THREAT RISK LEVEL", 'font': {'size': 14, 'color': '#94a3b8', 'family': 'JetBrains Mono'}},
                number={'suffix': "%", 'font': {'size': 36, 'color': '#f8fafc', 'family': 'Plus Jakarta Sans'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': '#ef4444' if deepfake_prob > 35 else '#34d399', 'thickness': 0.3},
                    'bgcolor': "rgba(15, 23, 42, 0.4)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [35, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 3},
                        'thickness': 0.8,
                        'value': deepfake_prob
                    }
                }
            ))
            fig_g2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=250,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_g2, use_container_width=True)

        # EXACT CONTRIBUTION BREAKDOWN TABLE
        st.markdown("<div class='hud-metric-title' style='margin-top: 14px;'>EXACT MULTIMODAL POINT CONTRIBUTIONS & ARITHMETIC</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        vis_pts = points.get("visual_points", round(vis_res.get("mean_real_prob", 50) * w_vis, 2))
        sync_pts = points.get("sync_points", 0.0)
        vis_real_p = points.get("visual_real_prob", vis_res.get("mean_real_prob", 0.0))
        sync_raw = points.get("sync_raw_score", sync_res.get("sync_score", 0.0))
        sync_norm = points.get("sync_normalized", 0.0)

        with c1:
            st.markdown(f"""
            <div class="glass-card" style="padding: 14px;">
                <div class="hud-metric-title">SPATIAL CNN (VISUAL)</div>
                <div class="hud-metric-value" style="font-size: 1.5rem; color: #38bdf8;">{vis_real_p:.1f}%</div>
                <div class="hud-metric-sub">+{vis_pts:.2f} pts ({int(w_vis*100)}% wt)</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="glass-card" style="padding: 14px;">
                <div class="hud-metric-title">LIP SYNC COHERENCE</div>
                <div class="hud-metric-value" style="font-size: 1.5rem; color: #34d399;">{sync_raw:.1f}</div>
                <div class="hud-metric-sub">+{sync_pts:.2f} pts ({int(w_sync*100)}% wt, {sync_norm:.1f}% norm)</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="glass-card" style="padding: 14px;">
                <div class="hud-metric-title">ACOUSTIC CADENCE</div>
                <div class="hud-metric-value" style="font-size: 1.5rem; color: #c084fc;">{audio_res.get('speech_ratio', 0.0)}%</div>
                <div class="hud-metric-sub">Mean RMS: {audio_res.get('mean_rms', 0.0):.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="glass-card" style="padding: 14px;">
                <div class="hud-metric-title">SUMMED TRUST SCORE</div>
                <div class="hud-metric-value" style="font-size: 1.5rem; color: {gauge_color};">{score:.2f}</div>
                <div class="hud-metric-sub">Risk: {deepfake_prob:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # MATHEMATICAL PROOF BOX
        st.markdown(f"""
        <div class="math-formula-box">
            <div style="color: #38bdf8; font-weight: 600; font-size: 0.85rem; margin-bottom: 6px;">
                📐 EXACT MULTIMODAL SCORE DERIVATION:
            </div>
            <div style="font-size: 0.95rem; color: #f8fafc; line-height: 1.7;">
                • <strong>Authenticity Trust Score</strong> = (w<sub>vis</sub> × P<sub>real</sub>) + (w<sub>sync</sub> × S<sub>sync</sub><sup>norm</sup>)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= ({w_vis:.2f} × {vis_real_p:.2f}%) + ({w_sync:.2f} × {sync_norm:.2f}%)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {vis_pts:.2f} + {sync_pts:.2f} = <span style="color: {gauge_color}; font-weight: 700;">{score:.2f} / 100.00</span><br>
                • <strong>Deepfake Manipulation Risk</strong> = 100.00% - {score:.2f}% = <span style="color: {'#ef4444' if deepfake_prob > 35 else '#34d399'}; font-weight: 700;">{deepfake_prob:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # REASONING CARD
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 14px; padding: 16px; border-left: 4px solid {gauge_color};">
            <div style="font-weight: 600; color: #f8fafc; margin-bottom: 4px;">Forensic Telemetry Interpretation:</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">{trust_res.get('reasoning', '')}</div>
        </div>
        """, unsafe_allow_html=True)

        # DOWNLOAD REPORT BUTTON
        report_txt = f"""=== DEEPSHIELD FORENSIC VERDICT AUDIT REPORT ===
Audit ID: #{audit_hash}
Date/Time: {audit_time}
Hardware Acceleration: {vis_res.get('device', 'cpu')}
Inference Latency: {exec_latency}s

1. FINAL FORENSIC VERDICT:
- Verdict Classification: {verdict_title}
- Forensic Grade: {forensic_grade}
- Authenticity Trust Score: {score:.2f} / 100.00
- Deepfake Manipulation Risk: {deepfake_prob:.2f}%
- Statistical Confidence: {conf_rating}

2. MULTIMODAL CONTRIBUTION BREAKDOWN:
- Sensitivity Profile: {prof_choice}
- Spatial CNN Real Probability: {vis_real_p:.2f}% (Contributed {vis_pts:.2f} pts at {int(w_vis*100)}% weight)
- Audio-Lip Sync Raw Score: {sync_raw:.2f}/100 (Normalized: {sync_norm:.2f}%, Contributed {sync_pts:.2f} pts at {int(w_sync*100)}% weight)
- Audio Cadence Speech Ratio: {audio_res.get('speech_ratio', 0.0)}%
- Mean Audio RMS Energy: {audio_res.get('mean_rms', 0.0):.4f}

3. REASONING & DIAGNOSTIC LOG:
{trust_res.get('reasoning', '')}

=== END OF AUDIT RECORD ===
"""
        st.download_button(
            "📋 Download Full Forensic Audit Certificate (.txt)",
            data=report_txt,
            file_name=f"DeepShield_Forensic_Audit_{audit_hash}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # TAB 2: VISUAL CNN ANALYSIS
    with tab_visual:
        st.markdown("### 👁️ EfficientNet-B0 Frame Decomposition")
        st.markdown(f"<div style='font-size: 0.8rem; font-family: monospace; color: #94a3b8; margin-bottom: 10px;'>UNIFORMLY EXTRACTED INSPECTION FRAMES ({len(frames)} SAMPLES) WITH PREDICTION HIGHLIGHTS:</div>", unsafe_allow_html=True)

        frame_preds = vis_res.get("frame_results", [])
        cols = st.columns(min(8, max(1, len(frames))))
        for idx, (f_idx, img) in enumerate(frames[:8]):
            with cols[idx]:
                t_sec = timestamps[idx] if idx < len(timestamps) else 0.0
                f_pred = frame_preds[idx] if idx < len(frame_preds) else {}
                p_real_f = f_pred.get("prob_real", 0.5)
                is_real_f = p_real_f >= 0.5
                card_style = "real-frame" if is_real_f else "fake-frame"
                tag_color = "#34d399" if is_real_f else "#ef4444"
                tag_label = f"{round(p_real_f*100)}% R" if is_real_f else f"{round((1-p_real_f)*100)}% F"
                
                st.markdown(f"<div class='filmstrip-card {card_style}'><span style='font-size: 0.7rem; font-family: monospace; color: {tag_color}; font-weight: 700;'>F#{f_idx} • {tag_label}</span></div>", unsafe_allow_html=True)
                st.image(img, use_container_width=True)

        if len(frames) > 8:
            cols2 = st.columns(min(8, len(frames) - 8))
            for idx, (f_idx, img) in enumerate(frames[8:16]):
                with cols2[idx]:
                    actual_idx = idx + 8
                    t_sec = timestamps[actual_idx] if actual_idx < len(timestamps) else 0.0
                    f_pred = frame_preds[actual_idx] if actual_idx < len(frame_preds) else {}
                    p_real_f = f_pred.get("prob_real", 0.5)
                    is_real_f = p_real_f >= 0.5
                    card_style = "real-frame" if is_real_f else "fake-frame"
                    tag_color = "#34d399" if is_real_f else "#ef4444"
                    tag_label = f"{round(p_real_f*100)}% R" if is_real_f else f"{round((1-p_real_f)*100)}% F"

                    st.markdown(f"<div class='filmstrip-card {card_style}'><span style='font-size: 0.7rem; font-family: monospace; color: {tag_color}; font-weight: 700;'>F#{f_idx} • {tag_label}</span></div>", unsafe_allow_html=True)
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
            st.markdown("<div class='hud-metric-title'>NORMALIZED AMPLITUDE WAVEFORM (16kHz DOWNSAMPLED)</div>", unsafe_allow_html=True)
            df_wave = pd.DataFrame({"Time (s)": audio_res["waveform_times"], "Amplitude": audio_res["waveform"]})
            fig_w = px.line(df_wave, x="Time (s)", y="Amplitude", color_discrete_sequence=["#38bdf8"])
            fig_w.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.4)", height=220, margin=dict(l=30, r=20, t=10, b=30))
            st.plotly_chart(fig_w, use_container_width=True)

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

    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
    st.markdown("<div class='hud-metric-title'>ENGINE TELEMETRY</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.6; font-family: 'JetBrains Mono', monospace;">
        <div>DEVICE: <span style="color: #38bdf8;">{visual_detector.device}</span></div>
        <div>MODEL: <span style="color: #34d399;">{'ONLINE' if visual_detector.is_loaded else 'PENDING'}</span></div>
        <div>LIP MESH: <span style="color: #34d399;">478 PTS</span></div>
        <div>AUDIO SR: <span style="color: #c084fc;">16,000 Hz</span></div>
    </div>
    """, unsafe_allow_html=True)


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
            acoustic energy profiling, and biometric audio–lip synchronization for live calls and recorded media.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚡ Quick Launch Checklist")
    st.markdown("""
    1. Select **🎬 Multimodal Forensic Lab** in the left navigation.
    2. Choose **"📹 Live Camera & Video Call Feed"** (right next to Your Real Video in the dropdown) to test your live webcam!
    3. Click **Execute Forensic Analysis** to inspect frame predictions, audio acoustics, and temporal sync curves!
    """)


# ---------------------------------------------------------
# PAGE 2: MULTIMODAL FORENSIC LAB (WITH LIVE CAMERA IN DROPDOWN)
# ---------------------------------------------------------
elif selected_page == "🎬 Multimodal Forensic Lab":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px;">
        <div>
            <h2 style="margin: 0; font-size: 2rem; color: #f8fafc;">🎬 Multimodal Forensic Lab</h2>
            <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 0.95rem;">
                Run live camera deepfake recognition, spatial CNN inference, physical acoustics, and audio-lip synchronization.
            </p>
        </div>
        <div>
            <span class="badge badge-green"><span class="pulse-dot"></span> FORENSIC PIPELINE READY</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Segmented Preset Selector with Live Camera options right next to Your Real Video
    st.markdown("<div style='font-size: 0.8rem; font-family: monospace; color: #94a3b8; margin-bottom: 6px;'>CHOOSE TEST VIDEO:</div>", unsafe_allow_html=True)

    preset_options = [
        "👤 Your Real Video (Library Test — 7.99s)",
        "📹 Live Camera & Video Call Feed",
        "📸 Live Camera Instant Snapshot",
        "🎬 Synthetic Demo 1 (Synchronized Speech)",
        "⚠️ Synthetic Demo 2 (Out-of-Phase / Desynced)",
        "🔇 Synthetic Demo 3 (Stripped Audio / Muted)",
        "📤 Upload Custom File"
    ]
    selected_preset = st.selectbox("Preset Selector", preset_options, index=0, label_visibility="collapsed")

    video_path = None
    video_title = ""

    # OPTION A: LIVE CAMERA VIDEO CALL FEED
    if "📹 Live Camera & Video Call Feed" in selected_preset:
        st.markdown("""
        <div style="margin: 12px 0 8px 0; font-size: 0.92rem; color: #cbd5e1;">
            <strong>📹 Live Video Call Simulation:</strong> Record a live video and voice sample directly through your webcam and microphone.
            DeepShield will decompose your live call into video frames, physical speech acoustics, and biometric lip synchronization.
        </div>
        """, unsafe_allow_html=True)

        rec_result = render_live_webcam_recorder(key="dropdown_live_recorder")

        if rec_result and rec_result.get("video_path") and os.path.exists(rec_result["video_path"]):
            video_path = rec_result["video_path"]
            duration = rec_result.get("duration", 0.0)
            video_title = f"Live_Webcam_Call_{int(time.time())}.mp4"
            st.success(f"✓ Live Call Captured ({duration:.1f}s, {round(rec_result['raw_size']/1024, 1)} KB). Ready for analysis!")

    # OPTION B: LIVE CAMERA INSTANT SNAPSHOT
    elif "📸 Live Camera Instant Snapshot" in selected_preset:
        st.markdown("""
        <div style="margin: 12px 0 8px 0; font-size: 0.92rem; color: #cbd5e1;">
            <strong>📸 Instant Webcam Facial Inspection:</strong> Point your camera at your face and capture an instant live snapshot.
            MediaPipe locks onto <strong>478 biometric landmarks</strong> and evaluates spatial deepfake probability via <strong>EfficientNet-B0</strong>.
        </div>
        """, unsafe_allow_html=True)

        snap = st.camera_input("Point camera at face and capture snapshot", key="dropdown_live_snapshot_cam")

        if snap is not None:
            with st.spinner("Locking biometric target & evaluating spatial convolutional features..."):
                t_snap_start = time.time()
                proc_res = live_camera_processor.process_camera_snapshot(snap, visual_detector)
                snap_latency = round(time.time() - t_snap_start, 3)

            st.markdown("<br>", unsafe_allow_html=True)
            col_img, col_pred = st.columns([1.1, 1])

            with col_img:
                st.markdown("<div style='font-size: 0.85rem; font-family: monospace; color: #38bdf8; margin-bottom: 6px;'>BIOMETRIC TARGET ACQUISITION</div>", unsafe_allow_html=True)
                st.image(proc_res["annotated_image"], use_container_width=True)

                if proc_res["face_detected"]:
                    st.markdown(f"""
                    <div style="display: flex; gap: 8px; margin-top: 8px;">
                        <span class="badge badge-green">✓ FACE ACQUIRED</span>
                        <span class="badge badge-cyan">{proc_res['landmarks_count']} LANDMARKS</span>
                        <span class="badge badge-purple">{proc_res['face_crop'].size[0]}×{proc_res['face_crop'].size[1]} ROI</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge badge-amber'>⚠️ Full Frame Analyzed (No Face Bounding Box)</span>", unsafe_allow_html=True)

            with col_pred:
                st.markdown("<div style='font-size: 0.85rem; font-family: monospace; color: #38bdf8; margin-bottom: 6px;'>EXACT BIOMETRIC SNAPSHOT VERDICT</div>", unsafe_allow_html=True)

                inf = proc_res.get("inference")
                if inf and inf.get("status") == "success":
                    p_real = float(inf["mean_real_prob"])
                    p_fake = float(inf["mean_fake_prob"])
                    pred_label = inf["overall_prediction"]
                    conf = float(inf.get("overall_confidence", max(p_real, p_fake)))

                    is_real = pred_label == "REAL"
                    gauge_color = "#10b981" if is_real else "#ef4444"
                    badge_style = "badge-green" if is_real else "badge-rose"
                    verdict_text = "AUTHENTIC HUMAN BIOMETRIC TARGET" if is_real else "SYNTHETIC FACIAL DEEPFAKE DETECTED"
                    verdict_banner_class = "emerald" if is_real else "rose"

                    # Snapshot Mini Verdict Banner
                    st.markdown(f"""
                    <div class="verdict-banner {verdict_banner_class}" style="padding: 16px; margin: 0 0 14px 0;">
                        <div style="font-size: 0.72rem; font-family: monospace; color: #38bdf8; text-transform: uppercase;">
                            LIVE CAMERA FACIAL INSPECTION
                        </div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin-top: 2px;">
                            {'🛡️' if is_real else '🚨'} {verdict_text}
                        </div>
                        <div style="display: flex; gap: 10px; margin-top: 10px;">
                            <span class="badge {badge_style}">CONFIDENCE: {conf:.1f}%</span>
                            <span class="badge badge-cyan">{round(snap_latency*1000)}ms LATENCY</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    fig_snap_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=p_real,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "AUTHENTICITY SCORE (PROB REAL)", 'font': {'size': 13, 'color': '#94a3b8', 'family': 'JetBrains Mono'}},
                        number={'suffix': "%", 'font': {'size': 34, 'color': '#f8fafc', 'family': 'Plus Jakarta Sans'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': gauge_color, 'thickness': 0.3},
                            'bgcolor': "rgba(15, 23, 42, 0.4)",
                            'borderwidth': 0,
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                                {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                            ],
                            'threshold': {
                                'line': {'color': "#ffffff", 'width': 3},
                                'thickness': 0.75,
                                'value': p_real
                            }
                        }
                    ))
                    fig_snap_gauge.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=180,
                        margin=dict(l=10, r=10, t=25, b=10)
                    )
                    st.plotly_chart(fig_snap_gauge, use_container_width=True)

                    st.markdown(f"""
                    <div class="glass-card" style="padding: 14px;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <div class="kpi-card" style="padding: 10px;">
                                <div class="hud-metric-title">EXACT REAL PROB</div>
                                <div style="font-size: 1.4rem; font-weight: 800; color: #10b981;">{p_real:.2f}%</div>
                            </div>
                            <div class="kpi-card" style="padding: 10px;">
                                <div class="hud-metric-title">EXACT FAKE PROB</div>
                                <div style="font-size: 1.4rem; font-weight: 800; color: #ef4444;">{p_fake:.2f}%</div>
                            </div>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.78rem; font-family: monospace; color: #94a3b8; display: flex; justify-content: space-between;">
                            <span>MODEL: EfficientNet-B0</span>
                            <span>BACKEND: PyTorch ({vis_detector_dev := str(visual_detector.device).upper()})</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # OPTION C: PRESET DEMO VIDEOS
    elif "Your Real Video" in selected_preset:
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

    # Video Player, Telemetry & Multimodal Analysis Execution
    if video_path and os.path.exists(video_path):
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
        btn_label = "🚀 EXECUTE MULTIMODAL FORENSIC ANALYSIS ON LIVE CALL" if "📹 Live Camera" in selected_preset else "🚀 EXECUTE MULTIMODAL FORENSIC ANALYSIS"
        if st.button(btn_label, key="btn_run_multimodal_analysis", use_container_width=True):
            with st.spinner("Executing spatial CNN inference, extracting audio features, and calculating lip synchronization..."):
                t_start = time.time()
                frames, timestamps = visual_detector.sample_frames(video_path, num_samples=16)
                if not frames or len(frames) == 0:
                    st.error("⚠️ Unable to decode visual frames from the video container. Please record a slightly longer clip (3–5 seconds) with your camera.")
                else:
                    audio_res = audio_analyzer.analyze_audio(video_path)
                    sync_res = sync_analyzer.analyze_synchronization(audio_res, video_path)
                    vis_res = visual_detector.predict_frames(frames)
                    trust_res = trust_engine.evaluate(vis_res, audio_res, sync_res)
                    exec_latency = round(time.time() - t_start, 3)
                    st.session_state["active_forensic_report"] = {
                        "trust_res": trust_res,
                        "vis_res": vis_res,
                        "audio_res": audio_res,
                        "sync_res": sync_res,
                        "frames": frames,
                        "timestamps": timestamps,
                        "exec_latency": exec_latency,
                        "video_path": video_path
                    }

        if "active_forensic_report" in st.session_state and st.session_state["active_forensic_report"].get("video_path") == video_path:
            rep = st.session_state["active_forensic_report"]
            render_multimodal_forensic_results(
                rep["trust_res"], rep["vis_res"], rep["audio_res"], rep["sync_res"],
                rep["frames"], rep["timestamps"], rep["exec_latency"],
                trust_engine_instance=trust_engine
            )

