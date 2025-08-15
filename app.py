# import streamlit as st
# import torch
# import torch.nn as nn
# import torchaudio
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# from io import BytesIO
# import os
# import shap

# # --- Model Architecture ---
# class AudioCNN(nn.Module):
#     def __init__(self):
#         super(AudioCNN, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 3, 1, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(7168, 128)
#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x

# # --- Helper Functions ---
# @st.cache_resource
# def load_model_and_explainer():
#     model = AudioCNN()
#     if os.path.exists("model.pth"):
#         model.load_state_dict(torch.load("model.pth", map_location='cpu'))
#     model.eval()

#     explainer = None
#     if os.path.exists('data/for_2_sec_train.pt'):
#         train_data = torch.load('data/for_2_sec_train.pt')
#         background = train_data['data'][np.random.choice(train_data['data'].shape[0], 20, replace=False)]
#         explainer = shap.DeepExplainer(model, background)
#     else:
#         st.warning("`data/for_2_sec_train.pt` not found. SHAP explainability will be disabled.")
#     return model, explainer

# def preprocess_audio(audio_bytes):
#     try:
#         waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))
#         if sample_rate != 16000:
#             waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
#         if waveform.shape[0] > 1:
#             waveform = torch.mean(waveform, dim=0, keepdim=True)
#         target_len = 2 * 16000
#         waveform = waveform[:, :target_len] if waveform.shape[1] > target_len else torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
#         mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)(waveform)
#         return torchaudio.transforms.AmplitudeToDB()(mel_spec), waveform
#     except Exception as e:
#         st.error(f"Error processing audio: {e}")
#         return None, None

# # --- Plotting ---
# def plot_waveform(waveform_data, sr):
#     fig, ax = plt.subplots(figsize=(12, 2))
#     plt.style.use('dark_background')
#     fig.patch.set_facecolor('#191924')
#     ax.set_facecolor('#191924')
#     librosa.display.waveshow(waveform_data, sr=sr, ax=ax, color='#e45735')
#     ax.set_title("Audio Waveform", color='white', fontsize=14)
#     plt.xticks(color='#a0a0a0'); plt.yticks(color='#a0a0a0')
#     for spine in ax.spines.values():
#         spine.set_color('#3a3a4c')
#     return fig

# def plot_spectrogram_and_shap(spectrogram_data, shap_values, sr):
#     plt.style.use('dark_background')

#     # Convert SHAP values to NumPy and reshape for image_plot
#     if torch.is_tensor(shap_values[0]):
#         shap_values_np = shap_values[0].detach().cpu().numpy()
#     else:
#         shap_values_np = np.array(shap_values[0])
#     if shap_values_np.ndim == 2:
#         shap_values_np = shap_values_np.reshape(1, *shap_values_np.shape, 1)

#     # Create figure for spectrogram
#     fig, ax = plt.subplots(figsize=(12, 4))
#     fig.patch.set_facecolor('#191924')
#     ax.set_facecolor('#191924')
#     librosa.display.specshow(spectrogram_data, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
#     ax.set_title("Audio Spectrogram", color='white')
#     for spine in ax.spines.values():
#         spine.set_color('#3a3a4c')

#     # SHAP overlay
#     try:
#         shap.image_plot(shap_values_np, show=False)
#     except Exception as e:
#         st.warning(f"Could not render SHAP plot: {e}")

#     return plt.gcf()

# # --- Streamlit UI ---
# st.set_page_config(page_title="A.V.A.D.", page_icon="🛡️", layout="wide")

# def load_css():
#     st.markdown("""
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
#         html, body, [class*="st-"], [class*="css-"] { font-family: 'Inter', sans-serif; }
#         .stApp { background-color: #191924; }
#     </style>
#     """, unsafe_allow_html=True)

# load_css()

# if 'analysis_log' not in st.session_state:
#     st.session_state.analysis_log = []

# model, explainer = load_model_and_explainer()

# with st.sidebar:
#     st.title("A.V.A.D.")
#     st.caption("AI Voice Anomaly Detector")
#     st.write("---")
#     st.info("This tool uses a trained neural network and SHAP explainability to perform spectral analysis.")
#     if not os.path.exists("model.pth"):
#         st.error("**SYSTEM OFFLINE**\nModel `model.pth` not found.")
#     else:
#         st.success("**SYSTEM ONLINE**\nModel loaded successfully.")
#     st.write("---")
#     st.subheader("Session Log")
#     if st.session_state.analysis_log:
#         for log_entry in reversed(st.session_state.analysis_log):
#             st.markdown(log_entry, unsafe_allow_html=True)
#     else:
#         st.caption("No files analyzed yet.")

# st.header("Threat Assessment Console")
# uploaded_file = st.file_uploader("Upload an audio file for analysis.", type=["wav","mp3","flac"], label_visibility="collapsed")

# if uploaded_file:
#     audio_bytes = uploaded_file.read()
#     spectrogram, waveform = preprocess_audio(audio_bytes)
#     if spectrogram is not None:
#         prediction = model(spectrogram.unsqueeze(0)).item()
#         confidence = prediction if prediction > 0.5 else 1 - prediction
#         result = "Deepfake" if prediction > 0.5 else "Real"

#         log_color = "#e45735" if result == "Deepfake" else "#28a745"
#         st.session_state.analysis_log.append(f"<font color='{log_color}'>**{result.upper()}**</font>: {uploaded_file.name} ({confidence:.1%})")

#         st.write("---")
#         col1, col2 = st.columns([3,2])
#         with col1:
#             st.subheader("Visual Analysis")
#             if explainer and result == "Deepfake":
#                 shap_values = explainer.shap_values(spectrogram.unsqueeze(0))
#                 st.pyplot(plot_spectrogram_and_shap(
#                     spectrogram.squeeze().numpy(),
#                     shap_values,
#                     16000
#                 ))
#             else:
#                 fig, ax = plt.subplots(figsize=(12,4))
#                 librosa.display.specshow(spectrogram.squeeze().numpy(), sr=16000, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
#                 ax.set_title("Audio Spectrogram", color='white')
#                 st.pyplot(fig)

#             st.pyplot(plot_waveform(waveform.numpy().flatten(), 16000))

#         with col2:
#             st.subheader("Analysis Report")
#             if result == "Deepfake":
#                 st.error("**STATUS: ANOMALY DETECTED**")
#                 st.metric(label="Confidence", value=f"{confidence:.2%}", delta="High Risk", delta_color="inverse")
#                 st.info("Red areas on the SHAP plot indicate regions the model found most indicative of a deepfake.")
#             else:
#                 st.success("**STATUS: AUTHENTIC**")
#                 st.metric(label="Confidence", value=f"{confidence:.2%}", delta="Low Risk", delta_color="normal")
#                 st.caption("Spectral signature is consistent with human vocal patterns.")

#             st.subheader("Audio Playback")
#             st.audio(audio_bytes)
# else:
#     st.info("System idle. Awaiting audio file for analysis.")
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
    """Loads the pre-trained model and SHAP explainer, caching them."""
    model = AudioCNN()
    model_path = "model.pth"
    if os.path.exists(model_path):
        try:
            # Load model state dictionary from the .pth file
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            st.sidebar.success("✅ **SYSTEM ONLINE**\nModel loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"❌ **SYSTEM OFFLINE**\nError loading model: {e}")
            return None, None # Return None if model loading fails
    else:
        st.sidebar.error(f"❌ **SYSTEM OFFLINE**\nModel `{model_path}` not found.")
        return None, None # Return None if model file is not found
    
    model.eval() # Set model to evaluation mode

    explainer = None
    data_path = 'data/for_2_sec_train.pt'
    if os.path.exists(data_path):
        try:
            train_data = torch.load(data_path)
            # Check if 'data' key exists and has enough samples for SHAP background
            if 'data' in train_data and train_data['data'].shape[0] >= 20:
                # Select a random subset of training data for SHAP background
                background = train_data['data'][np.random.choice(train_data['data'].shape[0], 20, replace=False)]
                explainer = shap.DeepExplainer(model, background)
            else:
                st.sidebar.warning("⚠️ Training data for SHAP is insufficient or malformed. SHAP explainability will be disabled.")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error loading SHAP training data: {e}. SHAP explainability will be disabled.")
    else:
        st.sidebar.warning(f"⚠️ `{data_path}` not found. SHAP explainability will be disabled.")
    return model, explainer

def preprocess_audio(audio_bytes):
    """
    Preprocesses the uploaded audio bytes into a mel spectrogram and waveform.
    Resamples to 16kHz, converts to mono, pads/truncates to 2 seconds, and computes mel spectrogram.
    """
    try:
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or truncate waveform to 2 seconds (32000 samples at 16kHz)
        target_len = 2 * 16000
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        
        # Convert waveform to NumPy for librosa plotting
        waveform_np = waveform.squeeze().numpy()

        # Compute Mel Spectrogram
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512)
        mel_spec = mel_spec_transform(waveform)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Add batch and channel dimensions for model input (1, 1, n_mels, n_frames)
        return mel_spec_db.unsqueeze(0), waveform_np, 16000 # Return sample rate
    except Exception as e:
        st.error(f"🚫 **Error processing audio:** {e}. Please try another file.")
        return None, None, None # Return None if processing fails

# --- Plotting ---
def set_plot_style(fig, ax):
    """Applies a consistent dark theme style to matplotlib figures."""
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#191924') # Background color for the figure
    ax.set_facecolor('#191924') # Background color for the axes
    ax.tick_params(colors='#a0a0a0') # Color for axis ticks
    ax.xaxis.label.set_color('#a0a0a0') # Color for x-axis label
    ax.yaxis.label.set_color('#a0a0a0') # Color for y-axis label
    for spine in ax.spines.values(): # Color for chart borders
        spine.set_color('#3a3a4c')

def plot_waveform(waveform_data, sr):
    """Plots the audio waveform."""
    fig, ax = plt.subplots(figsize=(10, 2))
    set_plot_style(fig, ax)
    librosa.display.waveshow(waveform_data, sr=sr, ax=ax, color='#e45735') # Accent color for waveform
    ax.set_title("Audio Waveform", color='white', fontsize=14)
    plt.tight_layout()
    return fig

def plot_spectrogram(spectrogram_data, sr, title="Audio Spectrogram"):
    """Plots the mel spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    set_plot_style(fig, ax)
    
    # Ensure spectrogram_data is a 2D NumPy array
    if torch.is_tensor(spectrogram_data):
        spectrogram_data = spectrogram_data.squeeze().numpy()
    
    librosa.display.specshow(spectrogram_data, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma') # Using 'magma' colormap
    ax.set_title(title, color='white')
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB') # Add a colorbar for dB scale
    plt.tight_layout()
    return fig

def plot_shap_overlay(spectrogram_data, shap_values, sr):
    """Plots the mel spectrogram with SHAP values overlaid as contours."""
    fig, ax = plt.subplots(figsize=(10, 4))
    set_plot_style(fig, ax)

    # Ensure spectrogram_data is 2D numpy array for plotting
    if torch.is_tensor(spectrogram_data):
        spec_np = spectrogram_data.squeeze().numpy()
    else:
        spec_np = spectrogram_data.squeeze()

    librosa.display.specshow(spec_np, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    
    # Convert SHAP values to NumPy and ensure correct dimensions for overlay
    # SHAP values typically come as (1, 1, H, W) for image data
    if torch.is_tensor(shap_values[0]):
        shap_values_np = shap_values[0].detach().cpu().numpy()
    else:
        shap_values_np = np.array(shap_values[0])
    
    # Ensure SHAP values are at least 2D (height, width) for plotting, squeeze extra dims
    shap_values_np = shap_values_np.squeeze()

    # Resize SHAP values to match spectrogram dimensions if they differ slightly
    if shap_values_np.shape != spec_np.shape:
        from skimage.transform import resize
        # Resize SHAP values to match the spectrogram's height and width
        resized_shap = resize(shap_values_np, spec_np.shape, anti_aliasing=True, preserve_range=True)
        shap_values_to_plot = resized_shap
    else:
        shap_values_to_plot = shap_values_np

    # Overlay SHAP explanation using contourf for filled contours
    # Using 'bwr' (blue-white-red) colormap to show positive/negative contributions
    contour = ax.contourf(
        np.linspace(0, spec_np.shape[1] * (512 / sr), spec_np.shape[1]), # X-axis (time) mapping for contours
        librosa.mel_frequencies(n_mels=128), # Y-axis (mel frequencies) mapping
        shap_values_to_plot,
        levels=np.linspace(shap_values_to_plot.min(), shap_values_to_plot.max(), 20), # Create 20 contour levels
        cmap='bwr', 
        alpha=0.5, # Semi-transparent overlay
        extend='both' # Extend colors beyond min/max levels
    )
    
    ax.set_title("Audio Spectrogram with SHAP Explanations", color='white')
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB') # Colorbar for spectrogram dB
    plt.colorbar(contour, ax=ax, format='%.2f', label='SHAP Value') # Separate colorbar for SHAP values
    plt.tight_layout()
    return fig

# --- Streamlit UI ---
st.set_page_config(
    page_title="A.V.A.D. - AI Voice Anomaly Detector",
    page_icon="🛡️",
    layout="wide", # Use wide layout for more content space
    initial_sidebar_state="expanded" # Keep sidebar open by default
)

def load_css():
    """Loads custom CSS for styling the Streamlit application."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="st-"], [class*="css-"] {
            font-family: 'Inter', sans-serif;
            color: #E0E0E0; /* Light gray text for readability */
        }
        .stApp {
            background-color: #12121D; /* Darker background */
        }
        h1, h2, h3, h4, h5, h6 { /* Main header font size and color */
            color: #BB86FC; /* Purple accent for headers */
            font-weight: 700;
        }
        .stButton>button {
            background-color: #3a3a4c;
            color: #E0E0E0;
            border-radius: 8px;
            border: 1px solid #4a4a5c;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease-in-out;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #4a4a5c;
            border-color: #5a5a6c;
            color: #FFFFFF;
        }
        .stFileUploader>div>div>p {
            color: #a0a0a0;
        }
        .stFileUploader>div>div>button {
            background-color: #e45735; /* Accent color for upload button */
            color: white;
            border: none;
            cursor: pointer;
        }
        .stFileUploader>div>div>button:hover {
            background-color: #ff6f4a;
        }
        .stFileUploader label {
            visibility: hidden; /* Hide default label to use custom styling */
        }
        .stFileUploader div[data-testid="stFileUploaderDropzone"] {
            background-color: #2a2a35;
            border: 2px dashed #4a4a5c;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
        }
        .st-emotion-cache-k3gpfz { /* Input label styles (e.g., for file uploader text) */
            color: #BB86FC;
            font-weight: 600;
        }
        .st-emotion-cache-1r6dm1b { /* Metric label */
            color: #a0a0a0;
            font-size: 0.9rem;
        }
        .st-emotion-cache-1xv3772 { /* Metric value */
            color: #FFFFFF;
            font-size: 2.2rem;
            font-weight: 700;
        }
        .st-emotion-cache-bkb580 { /* Metric delta (positive - green) */
            color: #28a745;
        }
        .st-emotion-cache-1ftrzl9 { /* Metric delta (negative/inverse - red) */
            color: #e45735;
        }
        .st-emotion-cache-1wivcwm { /* Info box */
            background-color: #2a2a35;
            color: #BB86FC;
            border-left: 5px solid #BB86FC;
        }
        .st-emotion-cache-1c0029q { /* Success box */
            background-color: #2a2a35;
            color: #28a745;
            border-left: 5px solid #28a745;
        }
        .st-emotion-cache-1s0z0h0 { /* Error box */
            background-color: #2a2a35;
            color: #e45735;
            border-left: 5px solid #e45735;
        }
        .st-emotion-cache-1h50x5 { /* Warning box */
            background-color: #2a2a35;
            color: #ffcc00;
            border-left: 5px solid #ffcc00;
        }
        .stTabs [data-testid="stTab"] {
            background-color: #2a2a35;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 3px;
            padding: 10px 15px;
            font-weight: 600;
            color: #a0a0a0;
            transition: all 0.2s ease-in-out;
        }
        .stTabs [data-testid="stTab"]:hover {
            background-color: #3a3a4c;
            color: #E0E0E0;
        }
        .stTabs [data-testid="stTab"][aria-selected="true"] {
            background-color: #191924; /* Active tab background */
            color: #e45735; /* Active tab text color */
            border-bottom: 3px solid #e45735; /* Active tab indicator */
        }
    </style>
    """, unsafe_allow_html=True)

load_css() # Apply custom CSS

# Initialize session state variables if they don't exist
if 'analysis_log' not in st.session_state:
    st.session_state.analysis_log = []
# Flag to indicate if a new file has been uploaded and analysis is pending
if 'pending_analysis' not in st.session_state:
    st.session_state.pending_analysis = False
# Stores the name of the last uploaded file to detect new uploads
if 'last_uploaded_file_name' not in st.session_state:
    st.session_state.last_uploaded_file_name = None
# Stores the results of the most recent analysis for display
if 'current_analysis_results' not in st.session_state:
    st.session_state.current_analysis_results = None

# Load model and explainer once when the app starts
model, explainer = load_model_and_explainer()

# --- Sidebar ---
with st.sidebar:
    st.image("https://www.freeiconspng.com/uploads/shield-icon-28.png", width=80) # Shield icon for branding
    st.markdown("<h1 style='text-align: center; color: #BB86FC;'>A.V.A.D.</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0a0a0;'>AI Voice Anomaly Detector</p>", unsafe_allow_html=True)
    st.markdown("---") # Separator

    st.markdown("### ℹ️ About A.V.A.D.")
    st.info("This application leverages a Convolutional Neural Network (CNN) to analyze audio files and detect anomalies, potentially indicating deepfake audio. It also utilizes **SHAP (SHapley Additive exPlanations)** to provide insights into *why* a particular detection was made, highlighting crucial spectral features.")
    st.markdown("---")

    st.markdown("### ⏳ Session Log")
    if st.session_state.analysis_log:
        for log_entry in reversed(st.session_state.analysis_log): # Display most recent logs first
            st.markdown(log_entry, unsafe_allow_html=True)
        # Button to clear the session log
        if st.button("Clear Log", help="Erase all previous analysis entries from the log."):
            st.session_state.analysis_log = []
            st.session_state.last_uploaded_file_name = None # Clear last file name on log clear
            st.session_state.current_analysis_results = None # Clear current results too
            st.rerun() # Rerun to update the sidebar and clear main display
    else:
        st.caption("No files analyzed yet.") # Message when log is empty

# --- Main Content ---
st.markdown("<h2 style='color: #E0E0E0;'>🛡️ Threat Assessment Console</h2>", unsafe_allow_html=True)
st.markdown("Upload an audio file (.wav, .mp3, .flac) below to initiate a spectral threat assessment.")

# File uploader widget
uploaded_file = st.file_uploader(
    "Upload an audio file for analysis.", 
    type=["wav", "mp3", "flac"], 
    label_visibility="hidden", # Hide default label for custom styling
    accept_multiple_files=False, # Only allow one file at a time
    key="audio_uploader" # Unique key to manage the uploader's state
)

# Logic to trigger analysis only for new uploads
# Check if a file was uploaded AND it's a new file (not the same one from a previous rerun)
if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_file_name:
    st.session_state.pending_analysis = True # Set flag to indicate analysis is needed
    st.session_state.last_uploaded_file_name = uploaded_file.name # Store the new file name

# Perform analysis if pending_analysis flag is True
if st.session_state.pending_analysis:
    if model is not None: # Ensure the model was loaded successfully
        with st.spinner(f"Analyzing '{uploaded_file.name}'... Please wait."):
            audio_bytes = uploaded_file.read()
            spectrogram, waveform_np, sr = preprocess_audio(audio_bytes)

        if spectrogram is not None:
            prediction = model(spectrogram).item() # Get prediction from the model
            confidence = prediction if prediction > 0.5 else 1 - prediction # Calculate confidence
            result = "Deepfake" if prediction > 0.5 else "Real" # Determine result

            log_color = "#e45735" if result == "Deepfake" else "#28a745" # Color for log entry
            st.session_state.analysis_log.append(
                f"<span style='color:{log_color};'>**{result.upper()}**</span>: `{uploaded_file.name}` (Confidence: **{confidence:.1%}**)"
            )
            
            # Store all results in session state for display after rerun
            st.session_state.current_analysis_results = {
                "spectrogram": spectrogram,
                "waveform_np": waveform_np,
                "sr": sr,
                "prediction": prediction,
                "confidence": confidence,
                "result": result,
                "audio_bytes": audio_bytes,
                "file_name": uploaded_file.name # Store filename for display consistency
            }
            st.session_state.pending_analysis = False # Reset flag as analysis is complete
            st.rerun() # Force a rerun to update sidebar log and display new results
        else:
            st.session_state.pending_analysis = False # Reset if processing failed
    else:
        st.error("Model not loaded. Cannot perform analysis.")
        st.session_state.pending_analysis = False # Reset if model isn't ready

# Display results only if analysis is not pending and results are available
if not st.session_state.pending_analysis and st.session_state.current_analysis_results is not None:
    results = st.session_state.current_analysis_results # Retrieve stored results
    spectrogram = results['spectrogram']
    waveform_np = results['waveform_np']
    sr = results['sr']
    prediction = results['prediction'] # Not directly used for display, but kept for context
    confidence = results['confidence']
    result = results['result']
    audio_bytes = results['audio_bytes']
    file_name = results['file_name'] # Use stored file name for consistent display

    st.markdown("---") # Separator

    # Use columns for a balanced layout
    col1, col2 = st.columns([3, 2]) # Wider column for visuals, narrower for report

    with col1:
        st.subheader("📊 Visual Analysis")
        # Use tabs for switching between Spectrogram and Waveform
        tab1, tab2 = st.tabs(["Spectrogram", "Waveform"])

        with tab1:
            # Display SHAP overlay only if explainer is available and result is Deepfake
            if explainer and result == "Deepfake":
                with st.spinner("Generating SHAP explanation..."):
                    try:
                        # Pass the 4D tensor to explainer.shap_values
                        shap_values = explainer.shap_values(spectrogram) 
                        st.pyplot(plot_shap_overlay(spectrogram.squeeze().numpy(), shap_values, sr))
                        st.info("💡 **SHAP Explanation**: Red areas on the spectrogram indicate features that strongly contributed to the 'Deepfake' prediction.")
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate SHAP plot: {e}. Displaying regular spectrogram.")
                        st.pyplot(plot_spectrogram(spectrogram.squeeze().numpy(), sr))
            else:
                st.pyplot(plot_spectrogram(spectrogram.squeeze().numpy(), sr))
                if result == "Real" and explainer:
                    st.info("SHAP explanations are typically shown for anomaly detections (Deepfakes).")
                elif not explainer:
                    st.warning("⚠️ SHAP explainability is disabled (training data not found or malformed).")

        with tab2:
            st.pyplot(plot_waveform(waveform_np, sr))

    with col2:
        st.subheader("📋 Analysis Report")
        if result == "Deepfake":
            st.error("🚨 **STATUS: ANOMALY DETECTED**") # Error message for deepfake
            st.metric(label="Confidence", value=f"{confidence:.2%}", delta="High Risk", delta_color="inverse")
            st.write("This audio file exhibits characteristics consistent with a generated or manipulated voice. Further investigation is recommended.")
            st.markdown("""
            **Possible Causes:**
            - AI-synthesized speech (Deepfake)
            - Voice modification software
            - Audio manipulation artifacts
            """)
        else:
            st.success("✅ **STATUS: AUTHENTIC**") # Success message for real audio
            st.metric(label="Confidence", value=f"{confidence:.2%}", delta="Low Risk", delta_color="normal")
            st.write("The spectral signature of this audio is consistent with human vocal patterns, indicating authenticity.")
            st.markdown("""
            **Findings:**
            - Natural vocal characteristics
            - Absence of synthetic artifacts
            """)
        
        st.subheader("🔊 Audio Playback")
        st.audio(audio_bytes, format='audio/wav') # Play the uploaded audio

else:
    # Display initial message if no analysis is pending and no results are stored
    if not st.session_state.pending_analysis and st.session_state.current_analysis_results is None:
        st.info("👆 Upload an audio file above to begin the analysis.")

st.markdown("---") # Footer separator

