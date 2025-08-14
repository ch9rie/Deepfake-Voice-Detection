import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
import os
import shap

# --- Model Architecture ---
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7168, 128)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# --- Helper Functions ---
@st.cache_resource
def load_model_and_explainer():
    model = AudioCNN()
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location='cpu'))
    model.eval()

    explainer = None
    if os.path.exists('data/for_2_sec_train.pt'):
        train_data = torch.load('data/for_2_sec_train.pt')
        background = train_data['data'][np.random.choice(train_data['data'].shape[0], 20, replace=False)]
        explainer = shap.DeepExplainer(model, background)
    else:
        st.warning("`data/for_2_sec_train.pt` not found. SHAP explainability will be disabled.")
    return model, explainer

def preprocess_audio(audio_bytes):
    try:
        waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        target_len = 2 * 16000
        waveform = waveform[:, :target_len] if waveform.shape[1] > target_len else torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)(waveform)
        return torchaudio.transforms.AmplitudeToDB()(mel_spec), waveform
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None

# --- Plotting ---
def plot_waveform(waveform_data, sr):
    fig, ax = plt.subplots(figsize=(12, 2))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#191924')
    ax.set_facecolor('#191924')
    librosa.display.waveshow(waveform_data, sr=sr, ax=ax, color='#e45735')
    ax.set_title("Audio Waveform", color='white', fontsize=14)
    plt.xticks(color='#a0a0a0'); plt.yticks(color='#a0a0a0')
    for spine in ax.spines.values():
        spine.set_color('#3a3a4c')
    return fig

def plot_spectrogram_and_shap(spectrogram_data, shap_values, sr):
    plt.style.use('dark_background')

    # Convert SHAP values to NumPy and reshape for image_plot
    if torch.is_tensor(shap_values[0]):
        shap_values_np = shap_values[0].detach().cpu().numpy()
    else:
        shap_values_np = np.array(shap_values[0])
    if shap_values_np.ndim == 2:
        shap_values_np = shap_values_np.reshape(1, *shap_values_np.shape, 1)

    # Create figure for spectrogram
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#191924')
    ax.set_facecolor('#191924')
    librosa.display.specshow(spectrogram_data, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set_title("Audio Spectrogram", color='white')
    for spine in ax.spines.values():
        spine.set_color('#3a3a4c')

    # SHAP overlay
    try:
        shap.image_plot(shap_values_np, show=False)
    except Exception as e:
        st.warning(f"Could not render SHAP plot: {e}")

    return plt.gcf()

# --- Streamlit UI ---
st.set_page_config(page_title="A.V.A.D.", page_icon="🛡️", layout="wide")

def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="st-"], [class*="css-"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #191924; }
    </style>
    """, unsafe_allow_html=True)

load_css()

if 'analysis_log' not in st.session_state:
    st.session_state.analysis_log = []

model, explainer = load_model_and_explainer()

with st.sidebar:
    st.title("A.V.A.D.")
    st.caption("AI Voice Anomaly Detector")
    st.write("---")
    st.info("This tool uses a trained neural network and SHAP explainability to perform spectral analysis.")
    if not os.path.exists("model.pth"):
        st.error("**SYSTEM OFFLINE**\nModel `model.pth` not found.")
    else:
        st.success("**SYSTEM ONLINE**\nModel loaded successfully.")
    st.write("---")
    st.subheader("Session Log")
    if st.session_state.analysis_log:
        for log_entry in reversed(st.session_state.analysis_log):
            st.markdown(log_entry, unsafe_allow_html=True)
    else:
        st.caption("No files analyzed yet.")

st.header("Threat Assessment Console")
uploaded_file = st.file_uploader("Upload an audio file for analysis.", type=["wav","mp3","flac"], label_visibility="collapsed")

if uploaded_file:
    audio_bytes = uploaded_file.read()
    spectrogram, waveform = preprocess_audio(audio_bytes)
    if spectrogram is not None:
        prediction = model(spectrogram.unsqueeze(0)).item()
        confidence = prediction if prediction > 0.5 else 1 - prediction
        result = "Deepfake" if prediction > 0.5 else "Real"

        log_color = "#e45735" if result == "Deepfake" else "#28a745"
        st.session_state.analysis_log.append(f"<font color='{log_color}'>**{result.upper()}**</font>: {uploaded_file.name} ({confidence:.1%})")

        st.write("---")
        col1, col2 = st.columns([3,2])
        with col1:
            st.subheader("Visual Analysis")
            if explainer and result == "Deepfake":
                shap_values = explainer.shap_values(spectrogram.unsqueeze(0))
                st.pyplot(plot_spectrogram_and_shap(
                    spectrogram.squeeze().numpy(),
                    shap_values,
                    16000
                ))
            else:
                fig, ax = plt.subplots(figsize=(12,4))
                librosa.display.specshow(spectrogram.squeeze().numpy(), sr=16000, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                ax.set_title("Audio Spectrogram", color='white')
                st.pyplot(fig)

            st.pyplot(plot_waveform(waveform.numpy().flatten(), 16000))

        with col2:
            st.subheader("Analysis Report")
            if result == "Deepfake":
                st.error("**STATUS: ANOMALY DETECTED**")
                st.metric(label="Confidence", value=f"{confidence:.2%}", delta="High Risk", delta_color="inverse")
                st.info("Red areas on the SHAP plot indicate regions the model found most indicative of a deepfake.")
            else:
                st.success("**STATUS: AUTHENTIC**")
                st.metric(label="Confidence", value=f"{confidence:.2%}", delta="Low Risk", delta_color="normal")
                st.caption("Spectral signature is consistent with human vocal patterns.")

            st.subheader("Audio Playback")
            st.audio(audio_bytes)
else:
    st.info("System idle. Awaiting audio file for analysis.")


