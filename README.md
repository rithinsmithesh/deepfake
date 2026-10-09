# 🛡️ DeepShield: Multimodal Deepfake Detection Framework

DeepShield is a college engineering project designed to evaluate videos for potential deepfake manipulations across three complementary forensic vectors:
1. **Spatial Feature Manipulation:** 2D CNN frame classification using **EfficientNet-B0**.
2. **Physical Acoustic Signal Decomposition:** Audio waveform, RMS speech energy envelope, and 13-band Mel-Frequency Cepstral Coefficients (MFCCs).
3. **Cross-Modal Consistency:** Temporal correlation between vocal acoustic energy and geometric lip aperture changes (landmarks 13 & 14) via **MediaPipe Face Mesh**.

---

## 📂 Project Architecture

```
DeepShield_Project/
├── app.py                     # Streamlit cybersecurity-themed dashboard application
├── requirements.txt           # Verified Python dependencies
├── README.md                  # Comprehensive technical documentation
├── DEMO_GUIDE.md              # 5-minute faculty demonstration walkthrough & Q&A
├── .gitignore                 # Cache and artifact exclusions
├── models/
│   ├── face_landmarker.task   # MediaPipe FaceLandmarker model asset (auto-bundled)
│   └── FINAL_EfficientNet_B0_FakeAVCeleb.pth  # Trained visual weights (place here)
├── src/
│   ├── __init__.py            # Package root
│   ├── visual_detector.py     # EfficientNet-B0 frame extraction, preprocessing, inference
│   ├── audio_analyzer.py      # FFmpeg 16kHz extraction, MFCCs, RMS speech activity
│   ├── sync_analyzer.py       # MediaPipe lip aperture tracking & Pearson correlation
│   ├── trust_engine.py        # Preliminary multimodal fusion Trust Score engine
│   ├── dataset_analytics.py   # Dataset analysis for FakeAVCeleb CSV experiments
│   └── sample_generator.py    # Generates built-in synthetic test clips for demonstration
├── assets/
│   └── samples/               # Built-in demo videos for instant testing
│       ├── sample_talking_face.mp4  (Synchronized audio & lip movements)
│       ├── sample_desync_face.mp4   (Desynchronized / out-of-phase speech)
│       └── sample_no_audio.mp4      (Edge case: video without audio track)
├── notebooks/                 # Preserved original experimental notebooks
│   ├── deepshield-model.ipynb
│   └── deepshield-member3-audio-synchronization.ipynb
└── results/                   # Preserved original experiment CSV results
    ├── full_dataset_synchronization_results.csv
    └── member3_ppt_summary.csv
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment
- Python 3.10 to 3.13 on macOS, Linux, or Windows.
- Apple Silicon (MPS), NVIDIA CUDA, or CPU compute.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 🧠 Visual Model Integration (`FINAL_EfficientNet_B0_FakeAVCeleb.pth`)

The visual detection backbone is an **EfficientNet-B0** convolutional neural network with a custom binary classification head (`0 = FAKE`, `1 = REAL`), trained for 30 epochs on the **FakeAVCeleb v1.2** dataset (recorded in `notebooks/deepshield-model.ipynb`).

### Placing Model Weights:
1. Copy your trained checkpoint file into the `models/` directory:
   ```bash
   cp /path/to/FINAL_EfficientNet_B0_FakeAVCeleb.pth models/
   ```
2. Or upload it directly via the **Model Checkpoint Management** panel in the Streamlit left sidebar.

### Transparent Scientific Handling:
- **When Checkpoint is Missing:** The application strictly avoids fabricating random probabilities. It displays verified Kaggle test benchmarks (99.75% accuracy, 97.35% F1 score across 3,234 test frames) and marks visual inference as *Pending Checkpoint Upload*.
- **When Checkpoint is Loaded:** Live PyTorch inference executes across sampled video frames, outputting per-frame softmax probabilities and aggregated predictions.

---

## 🔬 Multimodal Forensic Pipeline

### 1. Visual Feature Analysis
- Samples up to 16 uniformly distributed frames across video duration.
- Normalizes RGB frames with ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- Computes frame-level prediction distributions.

### 2. Physical Acoustic Extraction
- Extracts video audio stream into 16kHz mono WAV using bundled FFmpeg.
- Calculates Root Mean Square (RMS) energy envelope to delineate vocal activity.
- Generates 13-coefficient Mel-Frequency Cepstral Coefficients (MFCC) heatmaps and spectral centroids.
- *Notice:* Clearly designated as physical signal decomposition, not an autonomous AI voice-clone classifier.

### 3. Audio–Lip Synchronization (Member 3 Research)
- Tracks anatomical upper inner lip (landmark 13) and lower inner lip (landmark 14) using MediaPipe Face Mesh.
- Computes vertical lip aperture: $\Delta y = |y_{14} - y_{13}|$.
- Interpolates mouth motion to audio time points and normalizes to $[0, 1]$.
- Calculates Pearson correlation ($r$) and statistical significance ($p$-value) against binary vocal activity ($RMS > \overline{RMS}$).
- Generates Synchronization Score:
  $$\text{Sync Score} = \max(0, r) \times 100$$

### 4. Preliminary Multimodal Trust Score
When both modalities are active:
$$\text{Trust Score} = (0.75 \times P_{\text{visual\_real}}) + (0.25 \times \text{Normalized Sync Score})$$
Where raw sync score is normalized against empirical FakeAVCeleb real-video benchmarks ($\min(100, \frac{\text{Sync}}{25} \times 100)$).

---

## 📊 Dataset & Verified Results

### 1. FakeAVCeleb Audio-Lip Synchronization Benchmark (230 Videos)
Empirical data from `results/full_dataset_synchronization_results.csv`:

| Class | Count | Mean Correlation ($r$) | Median Correlation ($r$) | Mean Sync Score | Median Sync Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real** | 100 | **+0.0562** | **+0.0648** | **9.37** | **6.48** |
| **Fake** | 100 | **+0.0158** | **-0.0019** | **7.16** | **0.00** |
| **Fake Voice Only** | 30 | **+0.0425** | **+0.0667** | **9.51** | **6.67** |

*Critical Insight:* Real videos demonstrate a higher median correlation than synthetic videos. However, high standard deviation ($\sigma \approx 0.17$) and class overlap emphasize that linear correlation is an exploratory temporal cue that must be augmented with spatial CNN analysis.

### 2. EfficientNet-B0 Visual Model Benchmark (3,234 Test Frames)
- **Test Accuracy:** 99.75%
- **Precision:** 97.35%
- **Recall:** 97.35%
- **F1-Score:** 97.35%
- **Confusion Matrix:**
  - True Negatives (Fake Correct): 3,079
  - False Positives (Fake Misclassified as Real): 4
  - False Negatives (Real Misclassified as Fake): 4
  - True Positives (Real Correct): 147

---

## 🛡️ Implementation Status Matrix

| Component | Status | Implementation Details |
| :--- | :--- | :--- |
| **Web UI & Dashboard** | ✅ Complete | Streamlit cybersecurity dark theme with real-time telemetry |
| **Frame Sampler & Metadata** | ✅ Complete | OpenCV uniform frame sampling with resolution & FPS metrics |
| **EfficientNet-B0 Inference** | ✅ Complete | PyTorch pipeline; ready for checkpoint; verified benchmark fallbacks |
| **Audio Feature Extraction** | ✅ Complete | 16kHz WAV decomposition, MFCC heatmaps, RMS speech envelope |
| **Audio–Lip Sync Correlation** | ✅ Complete | MediaPipe landmarks 13/14 + Pearson $r$ correlation |
| **Multimodal Trust Score** | ✅ Complete | Explainable weighted fusion ($0.75 \times V + 0.25 \times S$) |
| **Edge-Case Resilience** | ✅ Complete | Handles missing audio, missing faces, corrupted streams |
| **Dataset Visualizations** | ✅ Complete | Interactive Plotly boxplots, histograms, confusion matrix |

---

## ⚖️ Ethical & Scientific Disclaimers
1. **Preliminary Research Heuristic:** The Trust Score is an illustrative college engineering formulation, not an officially certified legal evidence standard.
2. **Phonetic Limitations:** Certain phonemes (bilabials like 'm', 'b', 'p') require closed lips during vocalization; linear correlation is an exploratory approximation.
3. **No Standalone Audio Classifier:** Audio feature extraction measures physical acoustic dynamics, but does not train an autonomous neural voice-clone classifier.
