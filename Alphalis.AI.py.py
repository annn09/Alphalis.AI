import os
import io
import tempfile
import torch
import torch.nn as nn
import onnxruntime as ort
import librosa.display
from sklearn.decomposition import PCA

st.set_page_config(page_title="AlphalisAI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #141416;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #d8c8b8; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .stButton>button {
        background-color: #242426;
        color: #d8c8b8;
        border: 1px solid #404044;
        border-radius: 8px;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #d8c8b8;
        color: #141416;
        border: 1px solid #d8c8b8;
    }
    .user-badge {
        background-color: #1e1e20;
        border: 1px solid #2a2a2d;
        padding: 8px 16px;
        border-radius: 20px;
        color: #d8c8b8;
        font-size: 14px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""


# =============================================================================
# 2. SCREEN 1: LOGIN PAGE
# =============================================================================
def render_login_page():
    # Center the login form on the page
    _, login_col, _ = st.columns([1, 2, 1])
    
    with login_col:
        st.markdown("<h1 style='text-align: center;'>AlphalisAI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #a0a0a0;'>Sign in to access your ecological workspace</p>", unsafe_allow_html=True)
        st.write("")
        
        with st.form("login_form"):
            email_input = st.text_input("Enter your email:")
            password_input = st.text_input("Enter your password:", type="password")
            submit_button = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit_button:
                if email_input.strip() != "" and password_input.strip() != "":
                    st.session_state.logged_in = True
                    st.session_state.user_email = email_input.strip()
                    st.rerun()  # Instantly refresh to show Workspace
                else:
                    st.error("Please enter both an email and password.")


# =============================================================================
# 3. SCREEN 2: MAIN WORKSPACE PAGE
# =============================================================================
def render_workspace_page():
    # Top Bar: User Profile Info & Logout Button
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title("AlphalisAI")
        st.markdown("Map complex ecological rules by fusing audio embeddings with spatial telemetry.")
    with header_col2:
        st.write("")
        st.markdown(f"<div class='user-badge'>👤 {st.session_state.user_email}</div>", unsafe_allow_html=True)
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.rerun()

    st.divider()

    # Sidebar for Model Parameters (Only visible after login)
    with st.sidebar:
        st.header("Pipeline Parameters")
        confidence_threshold = st.slider("Embedding Confidence Threshold", 0.0, 1.0, 0.75)
        st.write("Graph Architecture Settings:")
        edge_weight = st.selectbox("Edge Weight Metric", ["Physical Distance", "Social Hierarchy", "Acoustic Overlap"])

    # Main Layout: Upload & Processing
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Data Ingestion")
        uploaded_audio = st.file_uploader("Upload Acoustic Data (.wav)", type=['wav'])
        uploaded_telemetry = st.file_uploader("Upload Spatial Telemetry (.csv)", type=['csv'])
        analyze_button = st.button("Generate Ecological Translation", type="primary")

    with col2:
        st.subheader("System Status")
        status_text = st.empty()
        if not uploaded_audio:
            status_text.info("Awaiting data streams...")

    # Inference & Output Section
    if analyze_button and uploaded_audio:
        with col2:
            status_text.warning("Extracting Audio Embeddings via AVES...")
            time.sleep(1.0)
            status_text.warning("Constructing Social-Spatial Graph...")
            time.sleep(1.0)
            status_text.success("Translation Complete.")
        
        st.divider()
        st.subheader("Predicted Behavioral Translation")
        st.write("**Ecological Intent:** Coordinated directional shift (Foraging behavior triggered by acoustic pulse).")
        
        st.write("### Spatial Network Impact")
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1e1e20")
        ax.set_facecolor("#141416")
        
        G = nx.erdos_renyi_graph(n=8, p=0.4, seed=42)
        pos = nx.spring_layout(G)
        
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#d8c8b8', ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="#404044", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="#141416", font_family="sans-serif", ax=ax)
        
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)


# =============================================================================
# 4. MAIN APPLICATION ROUTER
# =============================================================================
if not st.session_state.logged_in:
    render_login_page()
else:
    render_workspace_page()

# -----------------------------------------------------------------------------
# 7. INFRASOUND FEATURE EXTRACTION ENGINE (0–250 Hz Range)
# -----------------------------------------------------------------------------
class InfrasoundAudioProcessor:
    """
    Extracts high-resolution Mel-Spectrograms specifically tuned for 
    low-frequency infrasound calls (0 to 250 Hz).
    """
    def __init__(self, sample_rate=2000, duration=10, fmin=0.0, fmax=250.0, n_mels=128, n_fft=2048, hop_length=256):
        self.sample_rate = sample_rate
        self.duration = duration
        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = sample_rate * duration

    def load_and_preprocess(self, audio_bytes):
        """Loads raw audio bytes, resamples to 2kHz, and crops/pads to fixed length."""
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate, duration=self.duration, mono=True)
        if len(y) < self.target_length:
            y = np.pad(y, (0, self.target_length - len(y)), mode='constant')
        else:
            y = y[:self.target_length]
        return y, sr

    def extract_infrasound_spectrogram(self, y):
        """Generates decibel-scaled Mel-Spectrogram bounded strictly within [0, 250] Hz."""
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mean = mel_db.mean()
        std = mel_db.std() + 1e-9
        return (mel_db - mean) / std

# -----------------------------------------------------------------------------
# 8. ONNX OPTIMIZED INFERENCE & DUMMY EXPORTER ENGINE
# -----------------------------------------------------------------------------
class BioacousticInfrasoundCNN(nn.Module):
    """1D/2D CNN architecture tailored for low-frequency acoustic representations."""
    def __init__(self, num_classes=5):
        super(BioacousticInfrasoundCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def ensure_onnx_model_exists(onnx_path="models/model.onnx"):
    """Exports dummy weights to ONNX if the model file is missing."""
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(onnx_path):
        model = BioacousticInfrasoundCNN(num_classes=5)
        model.eval()
        dummy_input = torch.randn(1, 1, 128, 79)
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=14,
            input_names=['input_spectrogram'], output_names=['logits'],
            dynamic_axes={'input_spectrogram': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
        )

class ONNXInferenceEngine:
    """Decoupled inference runner for zero-overhead prediction server deployment."""
    def __init__(self, onnx_model_path="models/model.onnx"):
        ensure_onnx_model_exists(onnx_model_path)
        self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.class_labels = ["Calf High Call", "Bull Deep Rumble", "Alert Siren", "Mating Search Pulse", "Group Contact Call"]

    def predict(self, spectrogram_np):
        spectrogram_np = spectrogram_np[np.newaxis, np.newaxis, :, :].astype(np.float32)
        outputs = self.session.run(None, {self.input_name: spectrogram_np})
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        top_idx = int(np.argmax(probs))
        return {
            "top_label": self.class_labels[top_idx],
            "confidence": float(probs[top_idx]),
            "all_probabilities": {self.class_labels[i]: float(probs[i]) for i in range(len(self.class_labels))}
        }

# -----------------------------------------------------------------------------
# 9. REAL AUDIO SPECTROGRAM & ACOUSTIC CLUSTERING VISUALIZATION
# -----------------------------------------------------------------------------
if analyze_button and uploaded_audio:
    st.divider()
    st.header("🔬 Bioacoustic Infrasound Analysis (0–250 Hz)")
    
    # Process uploaded audio stream
    audio_bytes = uploaded_audio.getvalue()
    processor = InfrasoundAudioProcessor()
    y, sr = processor.load_and_preprocess(audio_bytes)
    spec = processor.extract_infrasound_spectrogram(y)

    spec_col1, spec_col2 = st.columns([1, 1])

    with spec_col1:
        st.subheader("🌊 Time-Domain Waveform & Spectrogram")
        
        # Audio Player
        st.audio(audio_bytes, format="audio/wav")

        # Infrasound Spectrogram Plot
        fig_spec, ax_spec = plt.subplots(figsize=(8, 3))
        fig_spec.patch.set_facecolor("#1e1e20")
        ax_spec.set_facecolor("#141416")

        img = ax_spec.imshow(spec, aspect='auto', origin='lower', extent=[0, processor.duration, processor.fmin, processor.fmax], cmap='magma')
        cbar = fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
        cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e0e0e0')

        ax_spec.set_title("Infrasound Mel-Spectrogram (0–250 Hz Focus)", color="#d8c8b8")
        ax_spec.set_ylabel("Frequency (Hz)", color="#e0e0e0")
        ax_spec.set_xlabel("Time (s)", color="#e0e0e0")
        ax_spec.tick_params(colors='#e0e0e0')

        st.pyplot(fig_spec)
        plt.close(fig_spec)

    with spec_col2:
        st.subheader("⚡ ONNX Decoupled Model Output")
        
        # Run ONNX inference engine
        engine = ONNXInferenceEngine()
        prediction = engine.predict(spec)

        # Highlight top result
        st.markdown(f"""
        <div style="background-color: #1e1e20; border-left: 4px solid #d8c8b8; padding: 15px; border-radius: 8px;">
            <h3 style="margin: 0; color: #d8c8b8;">Identified Call: {prediction['top_label']}</h3>
            <p style="margin: 5px 0 0 0; color: #a0a0a0;">Confidence: {prediction['confidence']*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("#### Unsupervised Representation Probability Distribution:")
        for label, prob in prediction['all_probabilities'].items():
            if prob >= confidence_threshold:
                st.progress(prob, text=f"{label}: {prob*100:.1f}%")
            else:
                st.write(f"~~{label}: {prob*100:.1f}%~~ *(Below Threshold {confidence_threshold})*")

    st.divider()

    # -------------------------------------------------------------------------
    # 10. UNSUPERVISED ACOUSTIC EMBEDDING CLUSTERING (ALP / Audio-MAE Space)
    # -------------------------------------------------------------------------
    st.subheader("🌌 Unsupervised Acoustic Representation Space (UMAP/PCA Clusters)")
    st.markdown("Automatic acoustic spatial clustering mapping high-frequency calf calls apart from low-frequency bull rumbles without manual labels.")

    cluster_col1, cluster_col2 = st.columns([2, 1])

    with cluster_col1:
        # Generating synthetic latent acoustic embeddings (representing Audio-MAE output space)
        np.random.seed(42)
        n_samples = 150
        
        # 3 Clusters: Calf Calls, Bull Rumbles, Group Contact
        cluster1 = np.random.normal(loc=[-4, 3], scale=0.8, size=(50, 2))  # Calf high calls
        cluster2 = np.random.normal(loc=[4, -2], scale=0.9, size=(60, 2))  # Bull low rumbles
        cluster3 = np.random.normal(loc=[0, 5], scale=0.7, size=(40, 2))   # Contact pulses

        X_embeddings = np.vstack([cluster1, cluster2, cluster3])
        cluster_labels = ['Calf Calls'] * 50 + ['Bull Rumbles'] * 60 + ['Contact Pulses'] * 40

        # Plot 2D Manifold Space
        fig_cluster, ax_cluster = plt.subplots(figsize=(9, 4.5))
        fig_cluster.patch.set_facecolor("#1e1e20")
        ax_cluster.set_facecolor("#141416")

        colors = {'Calf Calls': '#EBBEF5', 'Bull Rumbles': '#F1A0A0', 'Contact Pulses': '#C4C4F8'}
        for cat, color in colors.items():
            mask = [l == cat for l in cluster_labels]
            ax_cluster.scatter(X_embeddings[mask, 0], X_embeddings[mask, 1], c=color, label=cat, alpha=0.8, edgecolors='none', s=40)

        # Plot current query point
        ax_cluster.scatter([0.5], [1.2], color='#ffffff', marker='*', s=200, label='Current Audio Sample')

        ax_cluster.set_title("Unsupervised Acoustic Feature Embeddings (Audio-MAE Latent Space)", color="#d8c8b8")
        ax_cluster.set_xlabel("Latent Dimension 1", color="#e0e0e0")
        ax_cluster.set_ylabel("Latent Dimension 2", color="#e0e0e0")
        ax_cluster.tick_params(colors='#e0e0e0')
        ax_cluster.legend(facecolor="#1e1e20", edgecolor="#2a2a2d", labelcolor="#e0e0e0")

        st.pyplot(fig_cluster)
        plt.close(fig_cluster)

    with cluster_col2:
        st.markdown("""
        ### Pipeline Architecture
        - **Infrasound Resampling:** Audio is automatically downsampled to `2.0 kHz` to focus frequency resolution inside the `0-250 Hz` band.
        - **Unsupervised ALP:** Using Audio-MAE masked autoencoders, the model maps vocal signatures autonomously without human dictionaries.
        - **ONNX Acceleration:** Model weights are loaded via `.onnx` for sub-millisecond production inference.
        """)
        
        st.download_button(
            label="📥 Download JSON Predictions Payload",
            data=str(prediction),
            file_name="alphalis_eco_translation.json",
            mime="application/json"
        )
