import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
import tempfile

# 1. Model Architecture
class DeepfakeAudioDetector(nn.Module):
    def __init__(self):
        super(DeepfakeAudioDetector, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 64 channels * 10 height * 32 width = 20480
        self.fc1 = nn.Linear(20480, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Preprocessing Logic (Supports MP3, WAV, etc.)
def process_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        y, sr = librosa.load(tmp_path, sr=16000)
        target_len = 4 * 16000
        y = librosa.util.fix_length(y, size=target_len)
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=40)
        log_S = librosa.power_to_db(mel, ref=np.max)
        
        log_S = (log_S - log_S.min()) / (log_S.max() - log_S.min() + 1e-6)
        log_S = 2.0 * log_S - 1.0
        
        target_width = 128
        current_width = log_S.shape[1]
        if current_width < target_width:
            log_S = np.pad(log_S, ((0, 0), (0, target_width - current_width)), mode='constant', constant_values=-1.0)
        else:
            log_S = log_S[:, :target_width]

        return torch.from_numpy(log_S).unsqueeze(0).unsqueeze(0).float()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# 3. Streamlit Interface
st.set_page_config(page_title="Deepfake Detection Platform", layout="centered")

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Home", "Audio Detection", "Image Detection (Coming Soon)"])

if choice == "Home":
    st.title("🛡️ Deepfake Detection Hub")
    st.markdown("""
    Welcome to the unified Deepfake Detection platform. 
    Select a mode from the sidebar to verify the authenticity of your media.
    """)

elif choice == "Audio Detection":
    st.title("🎙️ Audio Authenticity Verifier")
    st.write("Upload any audio file (MP3, WAV, M4A) to verify its authenticity.")
    
    # Uploader comes FIRST
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "m4a"])
    
    # Display details ONLY if file exists
    if uploaded_file is not None:
        st.info(f"Successfully loaded: **{uploaded_file.name}**")
        st.audio(uploaded_file)
        
        if st.button("Run Analysis"):
            with st.spinner('Analyzing vocal frequencies...'):
                model = DeepfakeAudioDetector()
                # Load weights onto CPU
                model.load_state_dict(torch.load("deepfake_audio_model.pth", map_location='cpu'))
                model.eval()
                
                input_data = process_audio(uploaded_file)
                with torch.no_grad():
                    output = model(input_data)
                    prob = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output).item()
                
                confidence = prob[0][prediction].item() * 100
                if prediction == 0:
                    st.success(f"✅ **REAL VOICE** (Confidence: {confidence:.2f}%)")
                else:
                    st.error(f"🚨 **FAKE / DEEPFAKE** (Confidence: {confidence:.2f}%)")

elif choice == "Image Detection (Coming Soon)":
    st.title("🖼️ Image Detection")
    st.info("The image detection platform is currently under development.")
