# 🎓 DeepShield: Faculty Demonstration Master Guide

This guide gives you an exact, step-by-step walkthrough to present DeepShield to your evaluation panel with complete confidence, technical rigor, and scientific honesty.

---

## ⚡ Quick Launch Commands

Open your terminal, navigate to your project directory, and launch the app:

```bash
cd "/Users/hemanshmurahari/Desktop/DL Project"
streamlit run app.py
```

The application will start immediately at: `http://localhost:8501`.

---

## ⏱️ 5-Minute Faculty Demonstration Script

### Minute 0:00 – 1:00 | Project Motivation & System Overview (Page 1)
- **What to show:** Open **"🏠 Dashboard & Overview"**.
- **What to say:**
  > *"Good evening, respected professors. Today I am presenting **DeepShield**, a multimodal deepfake detection framework.
  > Modern generative deepfakes can fool single-modality detectors: visual CNNs can miss advanced diffusion blends, and audio detectors can miss realistic speech synthesis.
  > DeepShield tackles this through a multimodal forensic strategy:
  > 1. **Spatial feature inspection** using an EfficientNet-B0 convolutional network.
  > 2. **Physical acoustic decomposition** using FFmpeg and Librosa to analyze vocal energy and MFCC frequencies.
  > 3. **Cross-modal temporal coherence** using MediaPipe Face Mesh to measure the physical synchronization between mouth movement and vocal energy."*
- **Action:** Point out the **System Telemetry** panel in the left sidebar showing active hardware acceleration (Apple MPS/CPU) and the module verification badges.

---

### Minute 1:00 – 2:30 | Live Multimodal Video Analysis (Page 2)
- **What to show:** Click **"🎬 Multimodal Video Analysis"**.
- **What to do:**
  1. Select **"Choose Built-in Demo Video"** and pick **"Sample 1: Talking Face (Synchronized Audio + Speech Motion)"**.
  2. Show the video player and the extracted technical container metadata (Resolution, Framerate, Frame Count, Duration).
  3. Click **"Run Full Multimodal Inspection"**.
- **What to say:**
  > *"Here, our automated pipeline processes the video in under 2 seconds. Let us examine the four tabs:*
  > - *In **Visual CNN Analysis**, our system samples uniformly distributed frames across time. If the trained weights checkpoint is attached, it computes frame-level softmax predictions. Notice our interface is scientifically honest: if weights are awaiting upload, it displays verified benchmark metrics rather than fabricating fake predictions.*
  > - *In **Audio Acoustics**, we extract the 16kHz audio stream, computing the speech energy envelope and a 13-band Mel-Frequency Cepstral Coefficients (MFCC) heatmap.*
  > - *In **Audio-Lip Synchronization**, MediaPipe Face Mesh tracks anatomical inner-lip landmarks 13 and 14 to isolate mouth aperture. Our Pearson correlation engine aligns vocal energy with lip movement over time, generating a real-time Synchronization Score.*
  > - *In **Multimodal Trust Verdict**, all verified signals are synthesized into an explainable Trust Score using our weighted fusion model."*

---

### Minute 2:30 – 3:15 | Testing Edge Cases & Desynchronization (Page 2)
- **What to do:**
  1. Switch the sample selector to **"Sample 2: Desynchronized Face (Out-of-Phase / Dubbed Video)"** and click **Run Full Multimodal Inspection**.
  2. Show that the Sync Score drops to **0.0 / 100** with a negative Pearson correlation ($r \approx -0.55$).
  3. Switch to **"Sample 3: Edge Case Video (No Audio Track Container)"** and click run.
- **What to say:**
  > *"A critical test of any forensic tool is resilience against edge cases. When we run our desynchronized video—simulating an AI voice dub or spliced visual—our Pearson correlation turns negative, dropping the sync score to zero and flagging the mismatch.
  > When we test a muted video without an audio track, DeepShield handles it gracefully without crashing, explaining that synchronization is inactive due to missing audio and adapting the fusion formula dynamically."*

---

### Minute 3:15 – 4:15 | Experimental Dataset Results & Critical Nuances (Page 3)
- **What to show:** Click **"📊 Dataset & Experiments"**.
- **What to say:**
  > *"Our project is backed by empirical experiments on 230 videos from the benchmark **FakeAVCeleb v1.2** dataset (100 Real, 100 Fake, 30 Fake Voice Only).
  > As shown in our interactive box plots:
  > - Real videos demonstrated a higher median correlation (+0.0648) and higher median sync score (6.48) than Fake videos (median correlation -0.0019, median sync score 0.0).
  > - However, our research uncovered a vital scientific nuance: the standard deviation across all classes is relatively high ($\sigma \approx 0.17$), resulting in distribution overlap.
  > - Furthermore, 'Fake Voice Only' videos score almost identically to real videos because the original facial speech movements remain unaltered.
  > - For our visual model, EfficientNet-B0 achieved 99.75% accuracy and 97.35% F1 score across 3,234 test frames. Because the test set had a 20:1 imbalance of Fake to Real frames, monitoring Precision and Recall was essential to prove our model did not merely bias toward the majority class."*

---

### Minute 4:15 – 5:00 | Architecture, Limitations & Future Scope (Page 4) & Conclusion
- **What to show:** Click **"ℹ️ Architecture & About"**.
- **What to say:**
  > *"To summarize our implementation matrix:
  > - Spatial frame extraction and CNN inference are fully operational.
  > - Physical acoustic decomposition is fully operational.
  > - MediaPipe lip aperture tracking and Pearson synchronization are implemented.
  > - We have distinguished what is complete from preliminary research: our Trust Score is an exploratory weighted fusion heuristic, and our audio module extracts physical properties rather than claiming to be a standalone AI voice-clone discriminator.
  > - In future work, we plan to replace linear Pearson correlation with cross-attention multimodal transformers that learn joint audio-visual phoneme-viseme embeddings.
  > Thank you, and I welcome any questions from the panel."*

---

## 🎯 Likely Faculty Questions & Short, Technically Accurate Answers

#### Q1: "What model architecture did you use for visual deepfake detection?"
**Answer:**
> *"We used an **EfficientNet-B0** convolutional neural network pretrained on ImageNet. We replaced the final linear classification layer with a 2-class head (Class 0: FAKE, Class 1: REAL). Input frames are resized to 224×224 and normalized using standard ImageNet mean and standard deviation. We trained with Adam optimizer at a learning rate of 0.0001 using CrossEntropyLoss for 30 epochs on FakeAVCeleb v1.2 frames."*

#### Q2: "How does your audio-lip synchronization work mathematically?"
**Answer:**
> *"First, we extract a 16kHz mono audio track and compute the Root Mean Square (RMS) energy envelope using a 2048 frame length and 512 hop length. We binarize speech activity where RMS exceeds the mean.
> Concurrently, we run MediaPipe Face Mesh on video frames to extract upper inner lip landmark 13 and lower inner lip landmark 14. We calculate vertical aperture $\Delta y = |y_{14} - y_{13}|$, interpolate mouth motion to audio timestamps, normalize it to $[0, 1]$, and compute the Pearson correlation coefficient ($r$).
> The preliminary Synchronization Score is defined as $\text{Sync Score} = \max(0, r) \times 100$."*

#### Q3: "Why did Fake Voice Only videos have almost the same sync score as Real videos in your dataset?"
**Answer:**
> *"In FakeAVCeleb, 'Fake Voice Only' videos consist of genuine video footage paired with cloned or synthesized audio. Because the visual facial dynamics belong to an authentic speaker talking in natural rhythm, and the synthesized voice is matched to the speech transcript, the temporal cadence of syllables and lip openings still roughly aligns. This empirical result proves why temporal synchronization alone cannot detect voice-cloned deepfakes and why spatial visual CNN analysis must be combined in a multimodal architecture."*

#### Q4: "Why don't you have a visual prediction if the checkpoint file is missing?"
**Answer:**
> *"In accordance with strict scientific integrity, we refuse to simulate fake predictions using random numbers. DeepShield includes the complete PyTorch inference architecture; if the checkpoint file `FINAL_EfficientNet_B0_FakeAVCeleb.pth` is placed in `models/` or uploaded via our sidebar, live inference executes immediately. If it is missing, we honestly report checkpoint status and display our verified Kaggle test metrics."*

#### Q5: "What is your Trust Score formula and how is it justified?"
**Answer:**
> *"Our preliminary Trust Score is defined as:
> $$\text{Trust Score} = 0.75 \times P_{\text{visual\_real}} + 0.25 \times \text{Normalized Sync Score}$$
> We allocate 75% weight to visual CNN analysis because it achieved 97.35% F1-score on spatial artifacts, while temporal synchronization is an exploratory cue weighted at 25%. If a video lacks audio, the visual model receives 100% weight. We clearly document this as an exploratory research heuristic rather than a legally accredited biometric certainty."*

#### Q6: "What are the limitations of using Pearson correlation for lip synchronization?"
**Answer:**
> *"Pearson correlation assumes a linear relationship between mouth opening height and vocal volume. However, speech phonetics is non-linear: bilabial consonants like 'p', 'b', and 'm' require the lips to close completely even during vocalization, while vowels require an open mouth. A future improvement is to use a 3D-CNN or cross-attention Transformer (like SyncNet or LipSync-Transformer) to learn non-linear phoneme-to-viseme mappings."*

---

## 📋 What Genuinely Works in This Prototype

1. ✅ **Streamlit Application:** Launches reliably, responsive cybersecurity dark UI, multi-page routing.
2. ✅ **Video Container Processing:** Reads MP4, AVI, MOV, MKV, extracts FPS, duration, resolution, total frames.
3. ✅ **Uniform Frame Sampling:** Extracts and previews frames across the video timeline.
4. ✅ **EfficientNet-B0 PyTorch Pipeline:** Complete model structure, transforms, and inference logic.
5. ✅ **Audio Signal Extraction:** Bundled FFmpeg extracts 16kHz mono WAV from any video container.
6. ✅ **Acoustic Analysis:** Computes RMS vocal energy envelope, 13-band MFCC heatmap, waveform, and spectral centroid.
7. ✅ **Audio-Lip Synchronization:** MediaPipe FaceLandmarker tracks landmarks 13 & 14, interpolates signals, calculates Pearson $r$, $p$-value, and Sync Score.
8. ✅ **Edge Case Resilience:** Handled videos with no audio, desynchronized audio, and corrupted headers without crashes.
9. ✅ **Real Dataset Visualization:** Interactive Plotly charts for 230 FakeAVCeleb videos and verified 3,234 visual test frames.
10. ✅ **Built-in Demo Samples:** Instant test clips (`sample_talking_face.mp4`, `sample_desync_face.mp4`, `sample_no_audio.mp4`) ready for live demonstration.
