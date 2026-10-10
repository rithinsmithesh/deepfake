"""
Generate DeepShield Final Presentation Deck (.pptx)
Widescreen 16:9, modern dark theme styling, professional cards, metrics, and presenter notes.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path="DeepShield_Final_Presentation.pptx"):
    prs = Presentation()
    # Set 16:9 widescreen dimensions (13.333 x 7.5 inches)
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]  # completely blank

    # Color Palette
    BG_COLOR = RGBColor(11, 15, 25)         # Deep slate dark #0B0F19
    CARD_BG = RGBColor(21, 31, 50)          # Card panel #151F32
    CARD_BORDER = RGBColor(40, 56, 88)      # Card border #283858
    TEXT_WHITE = RGBColor(248, 250, 252)    # #F8FAFC
    TEXT_MUTED = RGBColor(148, 163, 184)    # #94A3B8
    CYAN_ACCENT = RGBColor(56, 189, 248)    # #38BDF8
    GREEN_ACCENT = RGBColor(16, 185, 129)   # #10B981
    AMBER_ACCENT = RGBColor(245, 158, 11)   # #F59E0B
    PURPLE_ACCENT = RGBColor(192, 132, 252) # #C084FC

    def apply_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
        return card

    def add_header(slide, tag_text, title_text, subtitle_text=""):
        # Tag Badge
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.4))
        tf = tx_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p_tag = tf.paragraphs[0]
        p_tag.text = tag_text.upper()
        p_tag.font.size = Pt(10)
        p_tag.font.bold = True
        p_tag.font.color.rgb = CYAN_ACCENT

        # Title
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE
        p_title.space_before = Pt(4)

        # Subtitle
        if subtitle_text:
            p_sub = tf.add_paragraph()
            p_sub.text = subtitle_text
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = TEXT_MUTED
            p_sub.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    apply_background(s1)

    # Decorative background card
    add_card(s1, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9), bg_color=RGBColor(15, 23, 42), border_color=RGBColor(56, 189, 248))

    t_box = s1.shapes.add_textbox(Inches(1.4), Inches(1.5), Inches(10.5), Inches(4.5))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "🛡️ DEEPSHIELD // MULTIMODAL FORENSICS"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = CYAN_ACCENT

    p1 = tf1.add_paragraph()
    p1.text = "DeepShield: Explainable Multimodal Deepfake\n& Voice Manipulation Forensic Detection Engine"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_WHITE
    p1.space_before = Pt(12)

    p2 = tf1.add_paragraph()
    p2.text = "Synergistic Triangulation of Spatial CNN Features, Acoustic Spectrograms & Biometric Lip Synchronization"
    p2.font.size = Pt(14)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_before = Pt(12)

    p3 = tf1.add_paragraph()
    p3.text = "Final Project Presentation | Deep Learning & Computer Vision\nDeveloped & Verified on FakeAVCeleb Dataset | Powered by EfficientNet-B0 + MediaPipe"
    p3.font.size = Pt(12)
    p3.font.color.rgb = GREEN_ACCENT
    p3.space_before = Pt(28)

    notes1 = s1.notes_slide.notes_text_frame
    notes1.text = ("Good morning/afternoon respected faculty and evaluators. Today we present DeepShield, "
                   "an explainable multimodal forensic system designed to combat the rising threat of hyper-realistic "
                   "deepfakes and AI voice cloning in live video communications. Unlike traditional systems that examine "
                   "only video frames or only audio waveforms in isolation, DeepShield cross-validates visual spatial artifacts, "
                   "acoustic speech properties, and biometric lip synchronization to produce a tamper-resistant Trust Score.")

    # -------------------------------------------------------------
    # SLIDE 2: EXECUTIVE SUMMARY & MOTIVATION
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    apply_background(s2)
    add_header(s2, "Context & Motivation", "The Deepfake Threat Landscape", "Why video-only and audio-only security models are no longer sufficient")

    cards_data_s2 = [
        ("🚨 The Emerging Threat", 
         "Generative AI models (Wav2Lip, DiffSVC, ElevenLabs, FaceSwap) have commoditized deepfake creation. Fraudsters exploit synthetic personas in live Zoom/Teams meetings, CEO fraud, KYC spoofing, and social engineering.",
         CYAN_ACCENT),
        ("⚠️ Unimodal Detection Failure", 
         "Video-only CNNs fail when video compression hides pixel artifacts. Audio-only classifiers fail on authentic audio paired with swapped faces. Attackers easily bypass single-channel security filters.",
         AMBER_ACCENT),
        ("🛡️ The DeepShield Paradigm", 
         "Multimodal Defense in Depth: Real-time spatial feature analysis + acoustic feature profiling + cross-modal biometric alignment (MediaPipe FaceLandmarker + Pearson Correlation).",
         GREEN_ACCENT)
    ]
    for i, (title, desc, color) in enumerate(cards_data_s2):
        left = Inches(0.8 + i * 3.98)
        add_card(s2, left, Inches(1.8), Inches(3.78), Inches(4.8))
        box = s2.shapes.add_textbox(left + Inches(0.25), Inches(2.05), Inches(3.28), Inches(4.3))
        tf = box.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(16)
        p_t.font.bold = True
        p_t.font.color.rgb = color
        
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = TEXT_MUTED
        p_d.space_before = Pt(14)

    s2.notes_slide.notes_text_frame.text = (
        "In this slide we establish the critical problem: synthetic media is becoming undetectable to the naked eye. "
        "More importantly, existing research heavily focuses on unimodal detection—either video or audio. "
        "Our contribution is demonstrating that combining modalities creates a mathematical defense that synthetic tools struggle to fool.")

    # -------------------------------------------------------------
    # SLIDE 3: PROBLEM STATEMENT & CORE OBJECTIVES
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    apply_background(s3)
    add_header(s3, "Problem Statement", "Research Questions & Project Objectives", "Guiding questions shaping our forensic system architecture")

    # Left card: Problem formulation
    add_card(s3, Inches(0.8), Inches(1.8), Inches(5.67), Inches(4.8))
    box_l = s3.shapes.add_textbox(Inches(1.05), Inches(2.05), Inches(5.17), Inches(4.3))
    tfl = box_l.text_frame
    tfl.word_wrap = True
    p = tfl.paragraphs[0]
    p.text = "🎯 Core Research Problems"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    bullets_l = [
        "How to detect subtle blending boundaries and diffusion noise across video frames without high computational overhead?",
        "How to verify whether spoken acoustic audio physically corresponds to visible biological mouth movement?",
        "How to mathematically handle incomplete or adversarial inputs (e.g. videos with stripped or muted audio tracks)?",
        "How to present forensic verdicts with clear explainability rather than an opaque black-box probability?"
    ]
    for b in bullets_l:
        p = tfl.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_before = Pt(10)

    # Right card: Key Objectives
    add_card(s3, Inches(6.86), Inches(1.8), Inches(5.67), Inches(4.8))
    box_r = s3.shapes.add_textbox(Inches(7.11), Inches(2.05), Inches(5.17), Inches(4.3))
    tfr = box_r.text_frame
    tfr.word_wrap = True
    p = tfr.paragraphs[0]
    p.text = "✅ Engineering Objectives Met"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    bullets_r = [
        "Trained EfficientNet-B0 on FakeAVCeleb achieving 99.75% visual test accuracy on 3,234 validation frames.",
        "Built 16kHz Librosa/FFmpeg audio pipeline extracting RMS Vocal Energy Dynamics & 13-band MFCC Spectrograms.",
        "Engineered real-time Lip Aperture tracking using MediaPipe FaceLandmarker (Inner Lip Landmarks 13 & 14).",
        "Formulated Explainable Multimodal Trust Engine (0.75 Visual + 0.25 Sync) with Missing-Modality Penalties.",
        "Delivered interactive web forensic platform deployed live to Streamlit Cloud with Apple Silicon MPS acceleration."
    ]
    for b in bullets_r:
        p = tfr.add_paragraph()
        p.text = "✔ " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_before = Pt(10)

    s3.notes_slide.notes_text_frame.text = (
        "Here we articulate our specific problem statement and project goals. Evaluators should note that we solved "
        "not only the visual detection problem, but also cross-modal biological synchronization, and solved the edge-case "
        "where an attacker maliciously strips the audio track to bypass sync checks.")

    # -------------------------------------------------------------
    # SLIDE 4: SYSTEM ARCHITECTURE & PIPELINE
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    apply_background(s4)
    add_header(s4, "System Design", "End-to-End Multimodal Pipeline", "Triangulating truth across parallel visual, acoustic, and biometric analysis streams")

    steps = [
        ("Step 1: Ingestion", "Upload MP4 / MOV / WebM\nVideo Container Telemetry\n16 Uniform Frame Sampler", CYAN_ACCENT),
        ("Step 2: Dual Processing", "Stream A: 224×224 Frames\nStream B: 16kHz Mono WAV\nStream C: FaceMesh Landmarks", PURPLE_ACCENT),
        ("Step 3: Feature Modeling", "EfficientNet-B0 Spatial CNN\nLibrosa 13-band MFCCs\nLandmarks 13 & 14 Aperture", AMBER_ACCENT),
        ("Step 4: Cross-Modal Sync", "Pearson Correlation (r)\nLip-Aperture vs RMS Energy\nSync Score = max(0, r) * 100", GREEN_ACCENT),
        ("Step 5: Trust Fusion", "Trust = 0.75·V + 0.25·S\nDegradation Penalties\nForensic Verdict Report", CYAN_ACCENT)
    ]

    for i, (title, desc, col) in enumerate(steps):
        left = Inches(0.8 + i * 2.38)
        add_card(s4, left, Inches(1.8), Inches(2.26), Inches(4.8))
        box = s4.shapes.add_textbox(left + Inches(0.15), Inches(2.05), Inches(1.96), Inches(4.3))
        tf = box.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = col
        
        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(11)
        pd.font.color.rgb = TEXT_MUTED
        pd.space_before = Pt(14)

    s4.notes_slide.notes_text_frame.text = (
        "This slide depicts our end-to-end data pipeline. An uploaded video is split into frames and an isolated audio stream. "
        "Frames flow to EfficientNet-B0 and MediaPipe. The audio flows to Librosa. Features from both streams are correlated, "
        "and synthesized in our Trust Engine to yield an explainable score between 0 and 100.")

    # -------------------------------------------------------------
    # SLIDE 5: PILLAR 1 - SPATIAL CONVOLUTIONAL DETECTION
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    apply_background(s5)
    add_header(s5, "Pillar 1: Spatial Vision", "Spatial Visual Deepfake Detection (EfficientNet-B0)", "High-throughput frame inspection capturing boundary artifacts and blending noise")

    # Card 1: Architecture
    add_card(s5, Inches(0.8), Inches(1.8), Inches(5.67), Inches(4.8))
    box_l = s5.shapes.add_textbox(Inches(1.05), Inches(2.05), Inches(5.17), Inches(4.3))
    tfl = box_l.text_frame
    tfl.word_wrap = True
    p = tfl.paragraphs[0]
    p.text = "🧠 Model Architecture & Training"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    bullets_p1 = [
        "Backbone: EfficientNet-B0 with compound scaling (depth, width, resolution).",
        "Pretrained on ImageNet-1K; custom classification head with Linear(1280, 2) outputting (0: Fake, 1: Real).",
        "Pre-processing: Uniform 16-frame temporal sampling, center cropping, resizing to 224×224 px.",
        "Normalization: Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225].",
        "Hardware Acceleration: Native PyTorch Apple Silicon MPS (Metal Performance Shaders) / CUDA support."
    ]
    for b in bullets_p1:
        p = tfl.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_before = Pt(10)

    # Card 2: Performance Metrics
    add_card(s5, Inches(6.86), Inches(1.8), Inches(5.67), Inches(4.8))
    box_r = s5.shapes.add_textbox(Inches(7.11), Inches(2.05), Inches(5.17), Inches(4.3))
    tfr = box_r.text_frame
    tfr.word_wrap = True
    p = tfr.paragraphs[0]
    p.text = "📊 Empirical Validation Metrics"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    metrics = [
        ("Test Accuracy", "99.75%", "(3,234 test frames from FakeAVCeleb)"),
        ("Precision", "100.0%", "(Zero false positives on authentic faces)"),
        ("Recall", "99.74%", "(3,075 of 3,083 fake frames detected)"),
        ("F1-Score", "99.87%", "(Balanced performance across classes)")
    ]
    for m_name, m_val, m_sub in metrics:
        p = tfr.add_paragraph()
        p.text = f"{m_name}: {m_val} {m_sub}"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE
        p.space_before = Pt(10)

    p_note = tfr.add_paragraph()
    p_note.text = "Why EfficientNet-B0? It delivers SOTA accuracy with only 5.3M parameters, allowing real-time video call processing with sub-50ms per-frame latency."
    p_note.font.size = Pt(11)
    p_note.font.color.rgb = CYAN_ACCENT
    p_note.space_before = Pt(16)

    s5.notes_slide.notes_text_frame.text = (
        "Here we discuss Pillar 1. We chose EfficientNet-B0 because of its compound scaling property—balancing accuracy "
        "against inference speed. On 3,234 test frames from FakeAVCeleb, the model achieved 99.75% accuracy. "
        "Sub-50ms frame inference makes it viable for real-time video call monitoring.")

    # -------------------------------------------------------------
    # SLIDE 6: PILLAR 2 - ACOUSTIC SPEECH PROFILING
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    apply_background(s6)
    add_header(s6, "Pillar 2: Acoustic Analysis", "Acoustic Signal Processing & Vocal Energy Profiling", "Decomposing speech into spectral envelopes and vocal tract dynamic representations")

    cards_s6 = [
        ("🔊 16kHz Stream Demuxing", 
         "Extracted via FFmpeg as a pristine 16,000 Hz 16-bit mono WAV stream.\n\nEliminates multi-channel phase noise and standardizes the sample rate for forensic acoustic extraction.",
         CYAN_ACCENT),
        ("📈 RMS Vocal Energy Envelope", 
         "Calculates Root Mean Square (RMS) frame-level vocal amplitude: RMS = sqrt(mean(x^2)).\n\nIdentifies active speech syllables vs. inter-speech pauses and background silence.",
         PURPLE_ACCENT),
        ("🌈 13-Band MFCC Spectrograms", 
         "Extracts Mel-Frequency Cepstral Coefficients (MFCCs) with 512 hop-length.\n\nCaptures the acoustic shape of the human vocal tract. Synthetic voice clones often display discontinuous higher-frequency harmonics.",
         GREEN_ACCENT)
    ]
    for i, (title, desc, color) in enumerate(cards_s6):
        left = Inches(0.8 + i * 3.98)
        add_card(s6, left, Inches(1.8), Inches(3.78), Inches(4.8))
        box = s6.shapes.add_textbox(left + Inches(0.25), Inches(2.05), Inches(3.28), Inches(4.3))
        tf = box.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(16)
        p_t.font.bold = True
        p_t.font.color.rgb = color
        
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = TEXT_MUTED
        p_d.space_before = Pt(14)

    s6.notes_slide.notes_text_frame.text = (
        "Pillar 2 handles audio acoustics. DeepShield standardizes audio to 16kHz mono using FFmpeg. "
        "We compute RMS vocal energy envelopes to track syllable dynamics, and 13-band MFCCs to capture vocal tract geometry. "
        "Synthetic vocoders often leave frequency artifacts in higher MFCC bands.")

    # -------------------------------------------------------------
    # SLIDE 7: PILLAR 3 - BIOMETRIC AUDIO-LIP SYNCHRONIZATION
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    apply_background(s7)
    add_header(s7, "Pillar 3: Cross-Modal Biometrics", "Audio–Lip Synchronization (MediaPipe + Pearson Correlation)", "Verifying that mouth mechanics physically correspond to acoustic phoneme bursts")

    add_card(s7, Inches(0.8), Inches(1.8), Inches(5.67), Inches(4.8))
    box_l = s7.shapes.add_textbox(Inches(1.05), Inches(2.05), Inches(5.17), Inches(4.3))
    tfl = box_l.text_frame
    tfl.word_wrap = True
    p = tfl.paragraphs[0]
    p.text = "👄 Biological Landmark Tracking"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PURPLE_ACCENT

    bullets_p3 = [
        "Tracks 478 3D facial mesh points via Google MediaPipe FaceLandmarker.",
        "Key Biological Anchor: Inner-lip vertical aperture between Landmark 13 (upper inner lip) and Landmark 14 (lower inner lip).",
        "Aperture Formula: D_lip(t) = sqrt((x13 - x14)^2 + (y13 - y14)^2) normalized by inter-pupillary distance.",
        "Temporal Interpolation: Aligns video frame rate (e.g. 30 FPS) with audio RMS frame rate (31.25 FPS) via linear interpolation.",
        "Robustness: Works under slight facial angle changes, head rotations, and diverse lighting conditions."
    ]
    for b in bullets_p3:
        p = tfl.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_before = Pt(10)

    add_card(s7, Inches(6.86), Inches(1.8), Inches(5.67), Inches(4.8))
    box_r = s7.shapes.add_textbox(Inches(7.11), Inches(2.05), Inches(5.17), Inches(4.3))
    tfr = box_r.text_frame
    tfr.word_wrap = True
    p = tfr.paragraphs[0]
    p.text = "📐 Pearson Correlation & Sync Score"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    bullets_p3_r = [
        "Pearson Correlation Coefficient (r):\nEvaluates the linear relationship between vocal energy E(t) and lip opening D(t).",
        "Sync Score Formulation:\nSync Score = max(0, r) * 100",
        "Dataset Ground Truth Context (FakeAVCeleb):\nAuthentic human speech averages Sync Score = 9.37 / 100 (median 6.48).",
        "Why isn't human sync 100%? Bilabial plosives ('M', 'B', 'P') force human lips to firmly close while vocal cord pressure is high!",
        "Desynced synthetic videos yield near-zero or negative correlation (r <= 0), triggering immediate detection."
    ]
    for b in bullets_p3_r:
        p = tfr.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED
        p.space_before = Pt(8)

    s7.notes_slide.notes_text_frame.text = (
        "Pillar 3 is our strongest cross-modal biometric weapon. We track the inner-lip aperture between MediaPipe landmarks 13 and 14. "
        "We align it with vocal volume and calculate the Pearson correlation r. An important academic finding: human speech naturally "
        "correlates around 9 to 35 on this scale because of bilabial consonants like M, B, and P where lips close during vocalization. "
        "Desynced videos have zero or negative correlation.")

    # -------------------------------------------------------------
    # SLIDE 8: DECISION ENGINE & TRUST FUSION FORMULATION
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    apply_background(s8)
    add_header(s8, "Decision Engine", "Explainable Multimodal Trust Score", "Unified mathematical formulation and missing-modality adversarial defense")

    # Formula Card
    add_card(s8, Inches(0.8), Inches(1.8), Inches(11.733), Inches(2.0), bg_color=RGBColor(15, 23, 42), border_color=CYAN_ACCENT)
    box_f = s8.shapes.add_textbox(Inches(1.1), Inches(1.95), Inches(11.1), Inches(1.7))
    tff = box_f.text_frame
    tff.word_wrap = True
    pf = tff.paragraphs[0]
    pf.text = "🔢 THE MULTIMODAL TRUST SCORE FORMULA"
    pf.font.size = Pt(12)
    pf.font.bold = True
    pf.font.color.rgb = CYAN_ACCENT

    pf2 = tff.add_paragraph()
    pf2.text = "Trust Score = 0.75 · V_real + 0.25 · S_norm"
    pf2.font.size = Pt(26)
    pf2.font.bold = True
    pf2.font.color.rgb = TEXT_WHITE
    pf2.space_before = Pt(6)

    pf3 = tff.add_paragraph()
    pf3.text = "Where: V_real = Mean Spatial CNN Real Probability (%) | S_norm = min(100, max(0, Sync Score × 2.0)) (%)"
    pf3.font.size = Pt(12)
    pf3.font.color.rgb = TEXT_MUTED
    pf3.space_before = Pt(4)

    # Lower Grid: Missing modality and thresholds
    add_card(s8, Inches(0.8), Inches(4.0), Inches(5.7), Inches(2.6))
    box_m = s8.shapes.add_textbox(Inches(1.05), Inches(4.15), Inches(5.2), Inches(2.3))
    tfm = box_m.text_frame
    tfm.word_wrap = True
    p = tfm.paragraphs[0]
    p.text = "⚠️ Missing-Modality Adversarial Guardrail"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = AMBER_ACCENT
    p_desc = tfm.add_paragraph()
    p_desc.text = ("• Attack Scenario: An adversary strips the audio track to avoid lip-sync detection.\n"
                  "• Defense: DeepShield voids the 25% sync credit. Trust drops from ~70% to 52.4% max.\n"
                  "• Status: Re-labeled as Inconclusive / Degraded (Audio Missing).\n"
                  "• Prevents attackers from gaming the system by omitting audio.")
    p_desc.font.size = Pt(11)
    p_desc.font.color.rgb = TEXT_MUTED
    p_desc.space_before = Pt(8)

    add_card(s8, Inches(6.833), Inches(4.0), Inches(5.7), Inches(2.6))
    box_t = s8.shapes.add_textbox(Inches(7.08), Inches(4.15), Inches(5.2), Inches(2.3))
    tft = box_t.text_frame
    tft.word_wrap = True
    p = tft.paragraphs[0]
    p.text = "🚦 Calibrated Forensic Decision Tiers"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p_tiers = tft.add_paragraph()
    p_tiers.text = ("🟢 [65 - 100] High Confidence Authentic: Natural facial features & coherent speech.\n"
                    "🟡 [45 - 65] Inconclusive / Caution: Degraded modality, low sync, or audio stripped.\n"
                    "🔴 [0 - 45] Synthetic / High Risk Deepfake: Significant spatial artifacting or desync.")
    p_tiers.font.size = Pt(11)
    p_tiers.font.color.rgb = TEXT_MUTED
    p_tiers.space_before = Pt(8)

    s8.notes_slide.notes_text_frame.text = (
        "Here is our mathematical fusion engine. We allocate 75% weight to Spatial CNN and 25% to Lip Sync. "
        "Importantly, we built an adversarial defense: if a video has no audio, we penalize the score by 25%, "
        "dropping a ~70% video to 52.4%, classifying it as Inconclusive. Attackers cannot cheat the system by muting clips.")

    # -------------------------------------------------------------
    # SLIDE 9: EXPERIMENTAL RESULTS & BENCHMARKS
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    apply_background(s9)
    add_header(s9, "Experimental Results", "Dataset Benchmarks on FakeAVCeleb v1.2", "Comprehensive empirical evaluation across 230 validation videos and 3,234 test frames")

    # Table of Results
    add_card(s9, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8))
    box_res = s9.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(11.1), Inches(4.4))
    tfr = box_res.text_frame
    tfr.word_wrap = True

    p = tfr.paragraphs[0]
    p.text = "📈 Quantitative Metric Summary"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    table_lines = [
        "Dataset: FakeAVCeleb v1.2 (Multi-ethnic celebrity videos generated via FaceSwap, FSGAN, Wav2Lip, SV2TTS)",
        "",
        "Metric Category                 Result              Benchmark Comparison / Significance",
        "─────────────────────────────────────────────────────────────────────────────────────────────",
        "Spatial Visual Accuracy         99.75%              3,226 correct out of 3,234 test frames",
        "Spatial Precision               100.0%              Zero false alarms on genuine human faces",
        "Spatial Recall                  99.74%              Detected 3,075 of 3,083 fake frames",
        "Authentic Sync Mean (r*100)     9.37 / 100          FakeAVCeleb real benchmark (median = 6.48)",
        "Synthetic Sync Mean             1.82 / 100          Desynced & Wav2Lip manipulated speech clips",
        "Pipeline Latency                1.8 - 2.5s          Full multimodal inference on 8-second video clip",
        "Hardware Mode                   MPS (Metal)         Apple Silicon GPU Acceleration (sub-50ms / frame)"
    ]
    for line in table_lines:
        p_l = tfr.add_paragraph()
        p_l.text = line
        p_l.font.size = Pt(11)
        p_l.font.name = "Courier New"
        p_l.font.color.rgb = TEXT_WHITE if "99." in line or "100." in line else TEXT_MUTED

    s9.notes_slide.notes_text_frame.text = (
        "Slide 9 displays our experimental benchmarks. FakeAVCeleb is the gold standard multi-modal deepfake dataset. "
        "Notice that our model achieved 99.75% accuracy. Crucially, the authentic sync score in the dataset averages 9.37, "
        "while synthetic desynced clips drop down to 1.82 or negative correlation.")

    # -------------------------------------------------------------
    # SLIDE 10: USER INTERFACE & DEMONSTRATION
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    apply_background(s10)
    add_header(s10, "System Implementation", "DeepShield Web Application & Live Deployment", "High-performance forensic dashboard built for security analysts and compliance officers")

    cards_s10 = [
        ("🎨 Cyber Forensic UI",
         "• Obsidian Dark Theme (#0B0F19)\n• Permanently pinned sidebar navigation\n• Radial speedometer Plotly Trust Gauge\n• Explainable signal contribution breakdown\n• Clean tabs for Verdict, CNN Frames, MFCC & Sync",
         CYAN_ACCENT),
        ("⚡ Processing Engine",
         "• PyTorch with Apple Silicon MPS GPU acceleration\n• OpenCV uniform temporal frame sampler\n• FFmpeg 16kHz audio extractor\n• MediaPipe FaceLandmarker pipeline\n• Sub-2.5 second total forensic execution time",
         PURPLE_ACCENT),
        ("🌐 Deployment Stack",
         "• Live on Streamlit Community Cloud\n• GitHub CI/CD automated deployment: github.com/rithinsmithesh/deepfake\n• Live URL: deepfakeaudio.streamlit.app\n• Debian Linux containerized dependencies (packages.txt + requirements.txt)",
         GREEN_ACCENT)
    ]
    for i, (title, desc, color) in enumerate(cards_s10):
        left = Inches(0.8 + i * 3.98)
        add_card(s10, left, Inches(1.8), Inches(3.78), Inches(4.8))
        box = s10.shapes.add_textbox(left + Inches(0.25), Inches(2.05), Inches(3.28), Inches(4.3))
        tf = box.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(16)
        p_t.font.bold = True
        p_t.font.color.rgb = color
        
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = TEXT_MUTED
        p_d.space_before = Pt(14)

    s10.notes_slide.notes_text_frame.text = (
        "We also placed heavy emphasis on engineering delivery. We built a production-grade Streamlit web app "
        "featuring an obsidian dark theme, interactive Plotly radial gauges, and deployed it live to the web. "
        "The entire codebase is containerized and available on GitHub.")

    # -------------------------------------------------------------
    # SLIDE 11: TEST CASE WALKTHROUGH
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    apply_background(s11)
    add_header(s11, "Validation Walkthrough", "Three Live Verification Scenarios", "Comparing system responses across authentic, desynced, and muted test videos")

    cases = [
        ("Scenario A: Real Talking Face", 
         "Video: Authentic human speech (7.99s)\nSpatial Real Prob: 69.89%\nLip Sync Score: 34.68 / 100\nPearson r: 0.3468 (Top 2% human tier)\n\nTrust Score: 69.75 / 100\nVerdict: High Confidence Authentic 🟢", 
         GREEN_ACCENT),
        ("Scenario B: Out-of-Phase / Desynced", 
         "Video: Synthetic Face / Desynced Audio\nSpatial Real Prob: 50.0%\nLip Sync Score: 0.00 / 100\nPearson r: -0.1420 (Anti-correlated)\n\nTrust Score: 37.5 / 100\nVerdict: Synthetic / Deepfake 🔴", 
         AMBER_ACCENT),
        ("Scenario C: Audio Stripped / Muted", 
         "Video: Muted Video (No Audio Track)\nSpatial Real Prob: 69.89%\nLip Sync Score: N/A (Nullified)\nPenalty: 25% Sync weight revoked\n\nTrust Score: 52.42 / 100\nVerdict: Inconclusive / Caution 🟡", 
         CYAN_ACCENT)
    ]
    for i, (title, desc, color) in enumerate(cases):
        left = Inches(0.8 + i * 3.98)
        add_card(s11, left, Inches(1.8), Inches(3.78), Inches(4.8))
        box = s11.shapes.add_textbox(left + Inches(0.25), Inches(2.05), Inches(3.28), Inches(4.3))
        tf = box.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = color
        
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = TEXT_MUTED
        p_d.space_before = Pt(12)

    s11.notes_slide.notes_text_frame.text = (
        "Here we walk evaluators through three concrete scenarios: First, an authentic video which scores 69.75 (Green Authentic). "
        "Second, a desynced deepfake where negative Pearson correlation drops the trust score to 37.5 (Red Deepfake). "
        "Third, a muted video where audio is stripped—our engine applies the missing modality penalty and drops it to 52.42 (Amber Caution).")

    # -------------------------------------------------------------
    # SLIDE 12: FUTURE SCOPE & CONCLUSION
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    apply_background(s12)
    add_header(s12, "Roadmap & Conclusion", "Summary of Contributions & Future Enhancements", "Transforming DeepShield into an enterprise video call security standard")

    add_card(s12, Inches(0.8), Inches(1.8), Inches(5.67), Inches(4.8))
    box_l = s12.shapes.add_textbox(Inches(1.05), Inches(2.05), Inches(5.17), Inches(4.3))
    tfl = box_l.text_frame
    tfl.word_wrap = True
    p = tfl.paragraphs[0]
    p.text = "🔮 Future Technical Roadmap"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PURPLE_ACCENT

    bullets_f = [
        "Spatiotemporal Transformers: Incorporating Video Swin or TimeSformer for cross-frame temporal motion consistency.",
        "Live WebRTC Browser Extension: Direct integration into Zoom, Google Meet, and Microsoft Teams streams.",
        "Synthetic Watermarking: Detection of generative diffusion latent signatures (e.g. Stable Video Diffusion).",
        "Edge Device Optimization: Quantization to INT8 via TensorRT and CoreML for embedded webcam deployment."
    ]
    for b in bullets_f:
        p = tfl.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_before = Pt(12)

    add_card(s12, Inches(6.86), Inches(1.8), Inches(5.67), Inches(4.8))
    box_r = s12.shapes.add_textbox(Inches(7.11), Inches(2.05), Inches(5.17), Inches(4.3))
    tfr = box_r.text_frame
    tfr.word_wrap = True
    p = tfr.paragraphs[0]
    p.text = "🏆 Key Takeaways & Conclusion"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    bullets_c = [
        "Multimodal beats unimodal: Combining spatial visual features with biometric lip-sync makes evasion exponentially harder.",
        "99.75% Visual Test Accuracy demonstrated on 3,234 FakeAVCeleb test frames.",
        "Biologically Grounded: Landmark-level lip aperture tracking provides verifiable physical evidence, not black-box guesses.",
        "Production-Grade: Sub-2.5s execution latency, open-source repository on GitHub, and live cloud deployment.",
        "Thank you! We welcome questions from the review committee."
    ]
    for b in bullets_c:
        p = tfr.add_paragraph()
        p.text = "✔ " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE if "Thank you" in b else TEXT_MUTED
        p.space_before = Pt(10)

    s12.notes_slide.notes_text_frame.text = (
        "In conclusion, DeepShield establishes that explainable multimodal deep learning is the future of synthetic media forensics. "
        "We achieved 99.75% visual accuracy, engineered biological lip-sync tracking, formulated a robust trust engine with adversarial defenses, "
        "and deployed a live web solution. Thank you for your time, and we are now ready for questions.")

    # Save presentation
    prs.save(output_path)
    print(f"✅ Successfully generated presentation deck: {output_path}")

if __name__ == "__main__":
    create_deck()
