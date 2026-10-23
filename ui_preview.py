import streamlit as st

st.set_page_config(
    page_title="VisionAI | Neural Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at top left, #102c4b 0%, transparent 42%),
        radial-gradient(ellipse at bottom right, #082b38 0%, transparent 38%),
        #050b16;
    color: #e6f7ff;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #e8faff !important;
}

.hero {
    padding: 32px;
    border: 1px solid #164b65;
    border-radius: 22px;
    background: linear-gradient(135deg, #101f35dd, #09232ddd);
    box-shadow: 0 0 30px #00d9ff12;
    margin-bottom: 24px;
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(28px, 4vw, 44px);
    font-weight: 700;
    letter-spacing: 2px;
    color: #53e8ff;
    text-shadow: 0 0 18px #00d9ff55;
}

.hero-subtitle {
    color: #9bb8ca;
    font-size: 15px;
    margin-top: 8px;
}

.panel {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #17435b;
    background: linear-gradient(145deg, #101d30ee, #091624ee);
    box-shadow: inset 0 1px 0 #ffffff08, 0 8px 28px #00000022;
    margin-bottom: 18px;
}

.metric-label {
    color: #8da9bd;
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.metric-value {
    font-family: 'Orbitron', sans-serif;
    color: #58eaff;
    font-size: 28px;
    margin-top: 10px;
}

div[data-testid="stFileUploader"] {
    padding: 14px;
    border: 1px dashed #24738c;
    border-radius: 15px;
    background: #0a1927;
}

.stButton > button {
    width: 100%;
    border: 1px solid #38dfff;
    border-radius: 12px;
    color: #03131d;
    font-weight: 700;
    background: linear-gradient(90deg, #48dcff, #65ffe0);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    box-shadow: 0 0 22px #36dfff66;
    border-color: #ffffff;
    transform: translateY(-1px);
}

section[data-testid="stSidebar"] {
    background: #07111e;
    border-right: 1px solid #16374c;
}

section[data-testid="stSidebar"] * {
    color: #d6edf7;
}

div[data-testid="stMetric"] {
    background: #0c1b2b;
    border: 1px solid #19465c;
    padding: 18px;
    border-radius: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #9bb8ca;
}

div[data-testid="stMetricValue"] {
    color: #53e8ff;
    font-family: 'Orbitron', sans-serif;
}

hr {
    border-color: #18394b;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 VISIONAI")
    st.caption("NEURAL IMAGE ANALYSIS LAB")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["🔬 Image Analyzer", "📊 Model Dashboard", "🧬 About VisionAI"],
    )

    st.divider()
    st.caption("SYSTEM STATUS")
    st.success("Interface initialized")
    st.caption("CNN prediction requires your existing model logic.")

# Main header
st.markdown("""
<div class="hero">
    <div class="hero-title">VISIONAI</div>
    <div class="hero-subtitle">
        NEURAL VISION LAB &nbsp; • &nbsp; DEEP LEARNING IMAGE CLASSIFICATION
    </div>
</div>
""", unsafe_allow_html=True)

if page == "🔬 Image Analyzer":
    st.subheader("Image Analysis")
    st.write("Upload an image to analyze it using your trained CNN model.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 📡 Input Console")
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            help="Choose a JPG or PNG image.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            from PIL import Image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Image loaded successfully", use_container_width=True)

            st.info(
                "Your upload is ready. Connect this section to your existing "
                "preprocessing and CNN prediction functions."
            )

    with right:
        st.markdown("""
        <div class="panel">
            <h3>◈ Prediction Console</h3>
            <p style="color:#91adbf">
                Prediction results will appear here after your existing
                model inference code runs.
            </p>
            <hr>
            <div class="metric-label">MODEL OUTPUT</div>
            <div class="metric-value">Awaiting Input</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
            <h3>📈 Confidence Analysis</h3>
            <p style="color:#91adbf">
                Display your existing top-3 class probabilities here.
            </p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 Model Dashboard":
    st.subheader("Model Dashboard")
    st.caption("Replace the example values below with your actual evaluation metrics.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "CIFAR-10")

    with col2:
        st.metric("Architecture", "CNN")

    with col3:
        st.metric("Training Status", "Trained")

    st.markdown("""
    <div class="panel">
        <h3>Training Overview</h3>
        <p style="color:#91adbf">
            Add your real test accuracy, training loss, validation metrics,
            and confusion matrix here. Avoid displaying sample metrics as
            actual model results.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🧬 About VisionAI":
    st.subheader("About the System")

    st.markdown("""
    <div class="panel">
        <h3>🧠 VisionAI — CNN Image Classifier</h3>
        <p style="color:#b0c9d8">
            VisionAI is an image classification interface built around a
            Convolutional Neural Network (CNN).
        </p>
        <hr>
        <p><b>Dataset:</b> CIFAR-10</p>
        <p><b>Task:</b> Image classification</p>
        <p><b>Framework:</b> PyTorch</p>
        <p><b>Interface:</b> Streamlit</p>
        <p><b>Workflow:</b> Image upload → preprocessing → CNN inference → results</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("VISIONAI • NEURAL LAB INTERFACE")