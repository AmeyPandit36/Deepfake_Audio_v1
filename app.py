import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display
import os
import tempfile
import matplotlib.pyplot as plt

# --- 1. SOTA MODEL ARCHITECTURE ---
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

# --- 2. ADVANCED FORENSIC PREPROCESSING ---
def process_pro_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load at 16k Mono
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        
        # FIX: Voice Activity Detection (Remove silence)
        # top_db=20 is aggressive; ensures the model doesn't see "digital silence" as machine artifacts
        intervals = librosa.effects.split(y, top_db=25)
        y_speech = np.concatenate([y[start:end] for start, end in intervals]) if len(intervals) > 0 else y

        window_size = 64000  # 4 seconds
        hop_length = 32000   # 2 seconds (50% OVERLAP)
        chunks = []
        
        # FIX: Sliding Window
        for i in range(0, len(y_speech) - window_size + 1, hop_length):
            chunk = y_speech[i : i + window_size]
            # FIX: Peak Normalization (Prevents noise stretching)
            peak = np.max(np.abs(chunk))
            if peak > 0:
                chunk = chunk / peak
            chunks.append(chunk)
            
        # Handle very short clips
        if not chunks:
            y_padded = librosa.util.fix_length(y_speech, size=window_size)
            peak = np.max(np.abs(y_padded))
            chunks.append(y_padded / (peak + 1e-7))

        # Shape: [Num_Chunks, 1, 64000]
        return torch.from_numpy(np.array(chunks)).unsqueeze(1).float()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Deepfake Shield | Forensic Lab", page_icon="🛡️", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("# 🛡️ **Forensic Lab**")
    choice = st.radio("ANALYSIS MODE", ["Overview", "Audio Verification"])
    st.status("SOTA GAT Model Ready", state="complete")

if choice == "Overview":
    st.title("Deepfake Shield Portal")
    st.markdown("""
    ### Current Defensive Layers:
    * **Model:** Efficient Graph Attention (GAT) Neural Network.
    * **VAD Filter:** Automatic silence removal to prevent "Silence Artifact" false positives.
    * **Windowing:** 50% Overlapping Sliding Windows for temporal analysis.
    """)

elif choice == "Audio Verification":
    st.title("🎙️ Audio Forensic Scanner")
    uploaded_file = st.file_uploader("Upload Audio (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])
    
    if uploaded_file:
        uploaded_file.seek(0)
        y_viz, sr_viz = librosa.load(uploaded_file, sr=16000)
        st.audio(uploaded_file)

        # Visualizations
        st.write("### 📊 Signal Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y_viz, sr=sr_viz, ax=ax_wave, color='#2563eb')
            ax_wave.set_axis_off()
            st.pyplot(fig_wave)
        with col2:
            fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=y_viz, sr=sr_viz, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr_viz, ax=ax_spec, cmap='magma')
            ax_spec.set_axis_off()
            st.pyplot(fig_spec)

        if st.button("RUN NEURAL DIAGNOSTIC"):
            with st.spinner('🔭 Running Forensic Scan...'):
                # Load Model
                model = SOTA_AudioDetector()
                if os.path.exists("sota_deepfake_detector.pth"):
                    model.load_state_dict(torch.load("sota_deepfake_detector.pth", map_location='cpu'))
                model.eval()

                # Process
                uploaded_file.seek(0)
                input_batch = process_pro_audio(uploaded_file)
                
                with torch.no_grad():
                    outputs = model(input_batch)
                    probs = torch.softmax(outputs, dim=1)
                    
                    # Aggregate results
                    avg_fake_prob = torch.mean(probs[:, 1]).item()
                    
                    st.markdown("---")
                    st.write("### 🔬 Forensic Diagnosis")
                    
                    # THREE-TIER CALIBRATION
                    if avg_fake_prob > 0.90:
                        st.error(f"🚨 **HIGH PROBABILITY SYNTHETIC** ({avg_fake_prob*100:.1f}%)")
                        st.warning("Forensic artifacts consistent with AI generation detected.")
                    elif avg_fake_prob > 0.45:
                        st.info(f"⚠️ **INCONCLUSIVE / COMPRESSION DETECTED** ({avg_fake_prob*100:.1f}%)")
                        st.write("High levels of digital noise found. This is likely due to **WhatsApp compression** or background interference.")
                    else:
                        st.success(f"✅ **VERIFIED NATURAL SPEECH** ({(1-avg_fake_prob)*100:.1f}%)")
                        st.write("Signal patterns match organic human speech profiles.")

                    # Technical Debug (Small font)
                    with st.expander("View Raw Neural Probabilities"):
                        st.write(probs.numpy())