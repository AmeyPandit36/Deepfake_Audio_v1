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
#         # Flattened size: 64 * 10 * 32 = 20480
#         self.fc1 = nn.Linear(64 * 10 * 32, 128)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 10 * 32)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class DeepfakeAudioDetector(nn.Module):
    def __init__(self):
        super(DeepfakeAudioDetector, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Use a dummy input to find the exact flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 40, 126) # 126 is common for 4s/16k/512hop
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flatten_size = x.numel() # Automatically counts the elements
            
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Use the dynamic size here
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Preprocessing Logic (Exact match to your extract_features function)
def process_audio(file):
    # Load audio at 16kHz
    y, sr = librosa.load(file, sr=16000)
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=50)
    # Fix length to 4 seconds
    target_len = 4 * 16000
    y = librosa.util.fix_length(y, size=target_len)
    
    # Generate Mel-spectrogram (40 mels)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=40)
    log_S = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize to [-1, 1]
    min_v, max_v = log_S.min(), log_S.max()
    if max_v > min_v:
        log_S = (log_S - min_v) / (max_v - min_v)
    else:
        log_S = np.zeros_like(log_S)
    log_S = 2.0 * log_S - 1.0
    
    # Convert to tensor and add batch/channel dims
    return torch.from_numpy(log_S).unsqueeze(0).unsqueeze(0).float()

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
    
    uploaded_file = st.file_uploader("Choose a wav file", type="wav")
    
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
