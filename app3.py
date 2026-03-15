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
# # 3. Streamlit Interface
# st.set_page_config(
#     page_title="Deepfake Shield | AI Forensic Tool",
#     page_icon="🛡️",
#     layout="centered"
# )

# # Custom CSS for a high-tech dark theme
# st.markdown("""
#     <style>
#     .main {
#         background-color: #0e1117;
#     }
#     .stButton>button {
#         width: 100%;
#         border-radius: 5px;
#         height: 3em;
#         background-color: #FF4B4B;
#         color: white;
#         font-weight: bold;
#         border: none;
#     }
#     .stButton>button:hover {
#         background-color: #ff3333;
#         border: 1px solid white;
#     }
#     .result-card {
#         padding: 20px;
#         border-radius: 10px;
#         border: 1px solid #30363d;
#         background-color: #161b22;
#         text-align: center;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Sidebar with better branding
# with st.sidebar:
#     st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
#     st.title("Forensic Suite")
#     choice = st.radio("Navigation", ["Home", "Audio Verifier", "Image Verifier"])
#     st.divider()
#     st.info("System Status: Online 🟢")

# if choice == "Home":
#     st.title("🛡️ Deepfake Shield")
#     st.subheader("Advanced Multi-Modal Media Authentication")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("""
#         ### **Capabilities**
#         * 🎙️ **Vocal Artifact Analysis**
#         * 🖼️ **GAN Pattern Detection**
#         * 📊 **Probability Scoring**
#         """)
#     with col2:
#         st.warning("**Note:** This is an AI research tool. Results should be verified by human forensic experts.")
    
#     st.image("https://miro.medium.com/v2/resize:fit:1400/1*H_O9Nq4T6A2Yh0p0U_j0wA.jpeg", caption="AI vs. Real Reality")

# elif choice == "Audio Verifier":
#     st.title("🎙️ Audio Forensic Analysis")
#     st.write("Detecting synthetic manipulation in vocal frequencies.")
    
#     uploaded_file = st.file_uploader("Upload Audio (MP3, WAV, M4A)", type=["wav", "mp3", "ogg", "m4a"])
    
#     if uploaded_file:
#         st.markdown('<div class="result-card">', unsafe_allow_html=True)
#         st.write(f"📂 **File:** {uploaded_file.name}")
#         st.audio(uploaded_file)
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         st.write("") # Spacer
        
#         if st.button("START DEEP SCAN"):
#             with st.spinner('🔬 Extracting Mel-Spectrogram and analyzing...'):
#                 model = DeepfakeAudioDetector()
#                 model.load_state_dict(torch.load("deepfake_audio_model.pth", map_location='cpu'))
#                 model.eval()
                
#                 input_data = process_audio(uploaded_file)
#                 with torch.no_grad():
#                     output = model(input_data)
#                     prob = torch.softmax(output, dim=1)
#                     prediction = torch.argmax(output).item()
                
#                 confidence = prob[0][prediction].item()
                
#                 # Big Result Visuals
#                 st.divider()
#                 if prediction == 0:
#                     st.balloons()
#                     st.success(f"### ✅ AUTHENTIC VOICE")
#                     st.progress(confidence)
#                     st.write(f"Confidence Level: **{confidence*100:.2f}%**")
#                 else:
#                     st.error(f"### 🚨 SYNTHETIC DEEPFAKE DETECTED")
#                     st.progress(confidence)
#                     st.write(f"AI Probability: **{confidence*100:.2f}%**")

# elif choice == "Image Verifier":
#     st.title("🖼️ Image Forensic Analysis")
#     st.info("Coming Soon: CNN-based Image Authenticator")
#     st.image("https://media.istockphoto.com/id/1310452331/vector/abstract-face-recognition-system.jpg?s=612x612&w=0&k=20&c=q11t8Nf_wQ3T0Yw1U6qW9z7z9z7z9z7z9z7z9z7z9z=")
#     st.info("The image detection platform is currently under development.")

# 3. Streamlit Interface
# 3. Streamlit Interface
st.set_page_config(
    page_title="Deepfake Shield | Forensic Analysis",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for a Clean, Scientific Light Theme
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
        color: #1e293b;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Professional Report Box */
    .report-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }

    /* Typography */
    h1, h2, h3 {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif;
    }

    /* Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# 🛡️ **Forensic Lab**")
    st.markdown("---")
    choice = st.radio("ANALYSIS MODE", ["Overview", "Audio Verification", "Image Analysis"])
    st.markdown("---")
    st.caption("v2.1.0-Stable")
    st.status("System Ready", state="complete")

if choice == "Overview":
    st.title("Deepfake Shield Portal")
    st.markdown("### Neural Network-Based Media Authentication")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        This platform utilizes advanced Convolutional Neural Networks (CNNs) to identify 
        discrepancies in digital media that are invisible to the human ear and eye.
        """)
        st.info("**Research Focus:** Identifying Mel-frequency artifacts in Generative AI speech models.")
    
    with col2:
        st.image("https://www.shutterstock.com/image-vector/artificial-intelligence-related-line-icon-600nw-1383838379.jpg", use_column_width=True)

elif choice == "Audio Verification":
    st.title("🎙️ Audio Forensic Scanner")
    st.write("Upload high-fidelity audio samples for deep-layer frequency analysis.")
    
    # File Uploader Container
    with st.container():
        uploaded_file = st.file_uploader("", type=["wav", "mp3", "m4a"])
        
    if uploaded_file:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown(f"**Target File:** `{uploaded_file.name}`")
        with col_b:
            st.audio(uploaded_file)
        
        if st.button("RUN NEURAL DIAGNOSTIC"):
            with st.spinner('Analyzing spectral consistency...'):
                # Core logic
                model = DeepfakeAudioDetector()
                model.load_state_dict(torch.load("deepfake_audio_model.pth", map_location='cpu'))
                model.eval()
                
                input_data = process_audio(uploaded_file)
                with torch.no_grad():
                    output = model(input_data)
                    prob = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output).item()
                
                confidence = prob[0][prediction].item()
                
                # Report UI
                st.markdown("---")
                if prediction == 0:
                    st.subheader("Result: ✅ VERIFIED HUMAN")
                    st.progress(confidence)
                    st.write(f"The system is **{confidence*100:.1f}%** confident this audio is authentic.")
                else:
                    st.subheader("Result: 🚨 SYNTHETIC ARTIFACTS DETECTED")
                    st.progress(confidence)
                    st.write(f"The system detected AI-generated signatures with **{confidence*100:.1f}%** probability.")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif choice == "Image Analysis":
    st.title("🖼️ Image Analysis")
    st.info("Module in Calibration: Integration with ResNet-50 expected in next update.")
