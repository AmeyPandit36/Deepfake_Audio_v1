import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os

# 1. Model Architecture (Exact match to your deepfakeaudio.ipynb)
# class DeepfakeAudioDetector(nn.Module):
#     def __init__(self):
#         super(DeepfakeAudioDetector, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
        
#         # This MUST be exactly 20480 (64 * 10 * 32) to match your .pth file
#         self.fc1 = nn.Linear(20480, 128)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
        
#         # Flatten the output for the linear layer
#         x = x.view(x.size(0), -1) 
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

 

# 2. Preprocessing Logic (Exact match to your extract_features function)
# def process_audio(file):
#     # Load audio at 16kHz
#     y, sr = librosa.load(file, sr=16000)
#     # Trim silence
#     y, _ = librosa.effects.trim(y, top_db=50)
#     # Fix length to 4 seconds
#     target_len = 4 * 16000
#     y = librosa.util.fix_length(y, size=target_len)
    
#     # Generate Mel-spectrogram (40 mels)
#     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=40)
#     log_S = librosa.power_to_db(mel, ref=np.max)
    
#     # Normalize to [-1, 1]
#     min_v, max_v = log_S.min(), log_S.max()
#     if max_v > min_v:
#         log_S = (log_S - min_v) / (max_v - min_v)
#     else:
#         log_S = np.zeros_like(log_S)
#     log_S = 2.0 * log_S - 1.0
    
#     # Convert to tensor and add batch/channel dims
#     return torch.from_numpy(log_S).unsqueeze(0).unsqueeze(0).float()


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
        # Force the flatten to match whatever comes out, 
        # but if it's not 20480, it will error here—helping us debug.
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# def process_audio(file):
#     # 1. Load at 16kHz
#     y, sr = librosa.load(file, sr=16000)
    
#     # 2. Fix length to EXACTLY 4 seconds (64000 samples)
#     # If the file is shorter, it pads; if longer, it crops.
#     target_len = 4 * 16000
#     y = librosa.util.fix_length(y, size=target_len)
    
#     # 3. Mel-spectrogram with specific parameters to hit the 20480 target
#     # n_mels=40 and the time-axis must result in 128 frames (128/4 = 32 after pooling)
#     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=501, n_mels=40)
#     # Note: hop_length=501 is a trick to ensure the width is exactly 128 frames
    
#     log_S = librosa.power_to_db(mel, ref=np.max)
    
#     # 4. Normalization
#     log_S = (log_S - log_S.min()) / (log_S.max() - log_S.min() + 1e-6)
#     log_S = 2.0 * log_S - 1.0
    
#     # 5. Ensure shape is [1, 1, 40, 128]
#     # If the width is 129 or 127, the linear layer will fail.
#     if log_S.shape[1] > 128:
#         log_S = log_S[:, :128]
#     elif log_S.shape[1] < 128:
#         log_S = np.pad(log_S, ((0,0), (0, 128 - log_S.shape[1])))

#     return torch.from_numpy(log_S).unsqueeze(0).unsqueeze(0).float()

import tempfile

def process_audio(uploaded_file):
    # 1. Save uploaded file to a temporary location
    # This allows librosa to read it regardless of format (mp3, wav, etc.)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 2. Load audio (Librosa handles the conversion to wav/16kHz automatically)
        y, sr = librosa.load(tmp_path, sr=16000)
        
        # 3. Fix length to 4 seconds
        target_len = 4 * 16000
        y = librosa.util.fix_length(y, size=target_len)
        
        # 4. Generate Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=40)
        log_S = librosa.power_to_db(mel, ref=np.max)
        
        # 5. Normalize
        log_S = (log_S - log_S.min()) / (log_S.max() - log_S.min() + 1e-6)
        log_S = 2.0 * log_S - 1.0
        
        # 6. Force width to 128 frames (to keep the 20480 shape)
        target_width = 128
        current_width = log_S.shape[1]
        if current_width < target_width:
            log_S = np.pad(log_S, ((0, 0), (0, target_width - current_width)), mode='constant', constant_values=-1.0)
        else:
            log_S = log_S[:, :target_width]

        return torch.from_numpy(log_S).unsqueeze(0).unsqueeze(0).float()
    
    finally:
        # Clean up the temp file after processing
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
    st.write("Upload a .wav file to check if it's a Deepfake or Real human voice.")
    st.write(f"Successfully loaded: **{uploaded_file.name}**")
    st.audio(uploaded_file)
    
    # uploaded_file = st.file_uploader("Choose a wav file", type="wav")
        # Change this line:
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "m4a"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Run Analysis"):
            with st.spinner('Analyzing vocal frequencies...'):
                # Load Model
                model = DeepfakeAudioDetector()
                # Ensure the .pth file is in your GitHub repo
                model.load_state_dict(torch.load("deepfake_audio_model.pth", map_location='cpu'))
                model.eval()
                
                # Predict
                input_data = process_audio(uploaded_file)
                with torch.no_grad():
                    output = model(input_data)
                    prob = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output).item()
                
                # Results
                confidence = prob[0][prediction].item() * 100
                if prediction == 0:
                    st.success(f"✅ REAL VOICE (Confidence: {confidence:.2f}%)")
                else:
                    st.error(f"🚨 FAKE / DEEPFAKE (Confidence: {confidence:.2f}%)")

elif choice == "Image Detection (Coming Soon)":
    st.title("🖼️ Image Detection")
    st.info("The image detection platform is currently under development. Stay tuned for the next update!")
