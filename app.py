import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
import tempfile

# 1. NEW SOTA Model Architecture (Matches Cell 4 of your notebook)
class EfficientGraphAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.norm(x + attn_output)

class SOTA_AudioDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, stride=5, padding=5) 
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=3, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.gat = EfficientGraphAttention(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        x = self.gat(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 2. NEW Preprocessing Logic (Raw Waveform)
def process_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load raw audio at 16kHz
        y, sr = librosa.load(tmp_path, sr=16000)
        # Fix length to 4 seconds (64,000 samples)
        y = librosa.util.fix_length(y, size=64000)
        # Z-score Normalization
        y = (y - np.mean(y)) / (np.std(y) + 1e-7)
        
        # Return as tensor with shape [1, 1, 64000]
        return torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- STREAMLIT UI CODE ---
st.set_page_config(page_title="Deepfake Shield | Forensic Lab", page_icon="🛡️", layout="wide")

# (Keep your existing CSS here...)

# Sidebar Navigation
with st.sidebar:
    st.markdown("# 🛡️ **Forensic Lab**")
    choice = st.radio("ANALYSIS MODE", ["Overview", "Audio Verification"])
    st.status("SOTA Model Loaded", state="complete")

if choice == "Overview":
    st.title("Deepfake Shield Portal")
    st.info("Now upgraded to SOTA AASIST-Lite Architecture for high-fidelity detection.")

elif choice == "Audio Verification":
    st.title("🎙️ Audio Forensic Scanner")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
    
    if uploaded_file:

        # Load audio for visualization (separate from model processing)
        y, sr = librosa.load(uploaded_file, sr=16000)
        
        st.audio(uploaded_file)

        # --- NEW VISUALIZATION SECTION ---
        st.write("### 📊 Signal Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Waveform** (Amplitude over Time)")
            fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#2563eb')
            ax_wave.set_axis_off() # Clean look
            st.pyplot(fig_wave)
            
        with col2:
            st.write("**Spectrogram** (Frequency over Time)")
            fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax_spec, cmap='magma')
            ax_spec.set_axis_off() # Clean look
            st.pyplot(fig_spec)
        # ---------------------------------
        
        if st.button("RUN NEURAL DIAGNOSTIC"):
            with st.spinner('Performing Deep Waveform Analysis...'):
                # Load the NEW weights
                model = SOTA_AudioDetector()
                model.load_state_dict(torch.load("sota_deepfake_detector.pth", map_location='cpu'))
                model.eval()
                
                input_data = process_audio(uploaded_file)
                with torch.no_grad():
                    output = model(input_data)
                    prob = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output).item()
                
                confidence = prob[0][prediction].item()
                
                if prediction == 0:
                    st.success(f"### ✅ VERIFIED HUMAN ({confidence*100:.1f}%)")
                else:
                    st.error(f"### 🚨 SYNTHETIC ARTIFACTS DETECTED ({confidence*100:.1f}%)")
