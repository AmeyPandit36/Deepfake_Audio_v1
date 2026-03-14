# 🛡️ Deepfake Shield: Multi-Modal Detection Platform

A Deep Learning-based platform designed to verify the authenticity of digital media. This project currently features a robust **Audio Deepfake Detector** and is built to expand into **Image Deepfake Analysis**.

## 🚀 Live Demo
**[Insert your Streamlit Link Here, e.g., https://audio-deepfake-check.streamlit.app]**

---

## 🎙️ Audio Detection (Current Phase)
The system uses a **Convolutional Neural Network (CNN)** to analyze vocal frequencies and detect synthetic artifacts that are invisible to the human ear.

### **Technical Specifications**
* **Architecture:** 4-Layer CNN (Conv2d, MaxPool2d, Linear).
* **Feature Extraction:** Mel-Spectrogram (40 Mel bands).
* **Input Standard:** 16kHz Mono Audio, 4-second duration.
* **Performance:** Achieved **90.70%+ accuracy** (and up to 100% on specific subsets) during validation.



---

## 🖼️ Image Detection (Upcoming)
The platform is architected to support image analysis using a vision-based CNN. This module will detect facial inconsistencies and GAN-generated pixel patterns.

---

## 🛠️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
