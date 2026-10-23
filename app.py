
import os
import io
import random
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader


# ============================================================
# VISIONAI — CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VisionAI | Image Intelligence",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "visionai_checkpoint.pth")
DATA_DIR = os.path.join(BASE_DIR, "data")

CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DARK BLUE UI
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: radial-gradient(ellipse at 10% 0%, #102a48 0%, transparent 38%),
                radial-gradient(ellipse at 90% 90%, #06303a 0%, transparent 34%), #050b16;
    color: #e6f7ff;
}
[data-testid="stSidebar"] { background: #07111f; border-right: 1px solid #16445a; }
[data-testid="stHeader"] { background: rgba(5, 11, 22, .88); }
h1, h2, h3 { color: #e8faff !important; letter-spacing: -.4px; }
p, label, li { color: #bfd1e1; }
.hero {
    padding: 30px 32px; border: 1px solid #1b6078; border-radius: 22px;
    background: linear-gradient(125deg, rgba(14,37,62,.96), rgba(7,23,36,.94));
    box-shadow: 0 0 32px rgba(0,217,255,.08), inset 0 1px 0 rgba(255,255,255,.04);
    margin-bottom: 26px;
}
.hero-tag, .section-label { color: #5be8ff; text-transform: uppercase; letter-spacing: 2px; font-size: 11px; font-weight: 800; }
.hero-title { font-family: 'Orbitron', sans-serif; font-size: clamp(27px, 3vw, 39px); line-height: 1.25; font-weight: 700; margin: 12px 0; color: #f4fdff; text-shadow: 0 0 20px rgba(42,220,255,.2); }
.hero-sub { color: #a8c4d6; font-size: 14px; line-height: 1.8; }
.metric-card, .result-card {
    padding: 22px; background: linear-gradient(145deg, rgba(15,31,49,.97), rgba(7,20,32,.97));
    border: 1px solid #1a5068; border-radius: 17px; min-height: 112px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.035), 0 8px 28px rgba(0,0,0,.16);
}
.metric-label { font-size: 11px; color: #8baec2; margin-bottom: 10px; letter-spacing: 1.5px; }
.metric-value { font-family: 'Orbitron', sans-serif; font-size: 25px; font-weight: 700; color: #59e8ff; text-shadow: 0 0 14px rgba(0,217,255,.17); }
.metric-note, .muted { color: #86a8bc; font-size: 12px; margin-top: 7px; }
.prediction-name { font-size: 30px; font-weight: 800; color: #fff; }
div.stButton > button {
    background: linear-gradient(90deg, #25c8f2, #5af3dc); color: #03121c;
    border: 1px solid #54eaff; border-radius: 11px; font-weight: 800; min-height: 44px;
    transition: .2s ease;
}
div.stButton > button:hover { box-shadow: 0 0 22px rgba(42,220,255,.32); border-color: #fff; color: #03121c; transform: translateY(-1px); }
[data-testid="stFileUploader"] { background: rgba(10,25,39,.9); border: 1px dashed #24728b; border-radius: 15px; padding: 12px; }
[data-testid="stMetric"] { background: rgba(13,29,45,.8); border: 1px solid #1b5068; padding: 16px; border-radius: 14px; }
[data-testid="stMetricValue"] { color: #5be8ff; font-family: 'Orbitron', sans-serif; }
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #24c9f3, #5af3dc); }
.stAlert { border-radius: 12px; }
hr { border-color: #1b3d50; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CNN ARCHITECTURE
# ============================================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# CIFAR-10 normalization
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])


# ============================================================
# DATASET AND TRAINING
# ============================================================

@st.cache_resource(show_spinner=False)
def get_test_dataset():
    return torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=EVAL_TRANSFORM,
    )


def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / max(total, 1), loss_sum / max(total, 1)


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    model = CNN().to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=DEVICE,
                weights_only=True,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            return model, checkpoint.get("history", []), True
        except (RuntimeError, KeyError, ValueError, TypeError):
            # Incompatible or invalid checkpoint: train this architecture.
            pass

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=TRAIN_TRANSFORM,
    )

    test_dataset = get_test_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = []

    for epoch in range(3):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / max(seen, 1)
        accuracy, test_loss = evaluate_model(model, test_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
        })

        # Displayed progress in the app
        st.session_state["training_status"] = (
            f"Epoch {epoch + 1}/3 completed — "
            f"test accuracy {accuracy:.2f}%"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        CHECKPOINT_PATH,
    )

    return model, history, False


# ============================================================
# APP HEADER AND SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🔷 VisionAI")
    st.caption("IMAGE INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Upload & Predict",
            "Accuracy Dashboard",
            "Prediction History",
            "About Model",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**MODEL STATUS**")
    st.caption(f"Compute device: `{DEVICE}`")
    st.caption("Dataset: CIFAR-10")
    st.caption("Architecture: CNN")

    if st.button("↻ Clear prediction history", use_container_width=True):
        st.session_state["prediction_history"] = []
        st.rerun()

st.markdown("""
<div class="hero">
    <div class="hero-tag">Deep Learning · Computer Vision</div>
    <div class="hero-title">VisionAI Image Intelligence</div>
    <div class="hero-sub">
        Classify images with a Convolutional Neural Network.
        Explore model performance, inspect confidence scores,
        and test your own images.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

if "training_status" not in st.session_state:
    st.session_state["training_status"] = ""

with st.spinner(
    "Loading saved model or training the CNN on CIFAR-10 "
    "for the first run..."
):
    model, history, loaded_checkpoint = load_or_train_model()

model.eval()

# Read checkpoint/test data for current evaluation metrics.
test_dataset = get_test_dataset()
test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)

if history:
    final_accuracy = history[-1]["test_accuracy"]
    final_loss = history[-1]["train_loss"]
else:
    final_accuracy, _ = evaluate_model(model, test_loader)
    final_loss = float("nan")


# ============================================================
# REUSABLE METRIC CARDS
# ============================================================

def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown("### Workspace overview")
    st.caption("Your model at a glance.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("TEST ACCURACY", f"{final_accuracy:.2f}%", "CIFAR-10 test set")

    with c2:
        metric_card("TRAINING EPOCHS", str(len(history)), "Recorded epochs")

    with c3:
        metric_card("IMAGE CLASSES", "10", "CIFAR-10 categories")

    with c4:
        metric_card("MODEL DEVICE", str(DEVICE).upper(), "PyTorch runtime")

    st.markdown("### Performance snapshot")

    if history:
        frame = pd.DataFrame(history).set_index("epoch")

        left, right = st.columns(2)

        with left:
            st.markdown("**Training loss**")
            st.line_chart(frame[["train_loss"]], color="#60A5FA")

        with right:
            st.markdown("**Test accuracy (%)**")
            st.line_chart(frame[["test_accuracy"]], color="#3B82F6")

    st.markdown("### Start exploring")

    a, b = st.columns(2)

    with a:
        st.markdown("**Image classification**")
        st.write("Upload a JPG, JPEG or PNG and view the top three predictions.")

    with b:
        st.markdown("**Model evaluation**")
        st.write("Review the recorded training loss and test accuracy.")


# ============================================================
# UPLOAD AND PREDICT
# ============================================================

elif page == "Upload & Predict":
    st.markdown("### Image classification")
    st.caption(
        "Supported classes: airplane, automobile, bird, cat, deer, "
        "dog, frog, horse, ship and truck."
    )

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="For best results, upload a clear image of one main object.",
    )

    if uploaded:
        image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")

        left, right = st.columns([1.05, 1])

        with left:
            st.image(image, caption=uploaded.name, use_container_width=True)

        input_tensor = transforms.Compose([
            transforms.Resize((32, 32)),
            EVAL_TRANSFORM,
        ])(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_values, top_indices = torch.topk(probabilities, k=3)

        predicted_index = top_indices[0].item()
        predicted_class = CLASSES[predicted_index]
        confidence = top_values[0].item() * 100

        with right:
            st.markdown('<div class="section-label">MODEL PREDICTION</div>',
                        unsafe_allow_html=True)
            st.markdown(f"## {predicted_class}")
            st.markdown(
                f'<div class="muted">Top-class confidence</div>'
                f'<div style="font-size:38px;font-weight:800;color:#60A5FA">'
                f'{confidence:.2f}%</div>',
                unsafe_allow_html=True,
            )
            st.progress(float(top_values[0].item()))

            st.markdown("#### Top-3 predicted classes")

            top_frame = pd.DataFrame({
                "Class": [CLASSES[i.item()] for i in top_indices],
                "Confidence (%)": [
                    round(v.item() * 100, 2) for v in top_values
                ],
            })

            st.bar_chart(
                top_frame.set_index("Class"),
                horizontal=True,
                color="#3B82F6",
            )

        # Record each upload once per file selection.
        selection_key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("last_selection_key") != selection_key:
            st.session_state["prediction_history"].insert(0, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded.name,
                "prediction": predicted_class,
                "confidence": round(confidence, 2),
            })
            st.session_state["last_selection_key"] = selection_key

        st.info(
            "This model recognises only CIFAR-10's ten categories. "
            "Confidence is the model's probability score, not a guarantee "
            "that the prediction is correct."
        )


# ============================================================
# ACCURACY DASHBOARD
# ============================================================

elif page == "Accuracy Dashboard":
    st.markdown("### Model performance")
    st.caption("Metrics recorded during the three-epoch training run.")

    if history:
        frame = pd.DataFrame(history)

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "FINAL TEST ACCURACY",
                f'{frame.iloc[-1]["test_accuracy"]:.2f}%',
                "Held-out CIFAR-10 test set",
            )

        with c2:
            metric_card(
                "FINAL TRAIN LOSS",
                f'{frame.iloc[-1]["train_loss"]:.4f}',
                "Cross-entropy objective",
            )

        with c3:
            metric_card(
                "EPOCHS RECORDED",
                str(len(frame)),
                "Training history",
            )

        st.markdown("#### Loss over epochs")
        st.line_chart(
            frame.set_index("epoch")[["train_loss"]],
            color="#60A5FA",
        )

        st.markdown("#### Test accuracy over epochs")
        st.line_chart(
            frame.set_index("epoch")[["test_accuracy"]],
            color="#3B82F6",
        )

        st.markdown("#### Recorded training metrics")
        st.dataframe(frame, use_container_width=True, hide_index=True)

        st.download_button(
            "Download metrics as CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="visionai_training_metrics.csv",
            mime="text/csv",
        )
    else:
        st.warning("No training history is available yet.")


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":
    st.markdown("### Prediction history")
    st.caption("Images classified in the current Streamlit session.")

    records = st.session_state["prediction_history"]

    if records:
        history_frame = pd.DataFrame(records)
        st.dataframe(
            history_frame,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download prediction history",
            data=history_frame.to_csv(index=False).encode("utf-8"),
            file_name="visionai_prediction_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No predictions yet. Open Upload & Predict to get started.")


# ============================================================
# ABOUT MODEL AND DATASET
# ============================================================

elif page == "About Model":
    st.markdown("### About VisionAI")
    st.write(
        "VisionAI is a deep learning image classification application "
        "that uses a Convolutional Neural Network (CNN) to classify "
        "images into the ten CIFAR-10 categories."
    )

    st.markdown("#### Dataset")

    st.write(
        "CIFAR-10 contains 60,000 colour images of size 32 × 32 pixels. "
        "The standard split contains 50,000 training images and "
        "10,000 test images across ten balanced classes."
    )

    st.markdown("#### CNN architecture")

    st.code(
        "Input RGB image (32 × 32)\n"
        "    ↓\n"
        "Convolution → Batch Normalization → ReLU → Max Pooling\n"
        "    ↓\n"
        "Convolution → Batch Normalization → ReLU → Max Pooling\n"
        "    ↓\n"
        "Convolution → Batch Normalization → ReLU → Max Pooling\n"
        "    ↓\n"
        "Flatten → Fully Connected Layer → Dropout\n"
        "    ↓\n"
        "10 output logits → Softmax probabilities",
        language="text",
    )

    st.markdown("#### Training configuration")

    st.dataframe(
        pd.DataFrame({
            "Parameter": [
                "Framework", "Optimizer", "Loss function",
                "Batch size", "Epochs", "Learning rate",
                "Evaluation",
            ],
            "Value": [
                "PyTorch", "Adam", "Cross-entropy",
                "128", "3", "0.001",
                "Held-out CIFAR-10 test set",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "This model is intended for academic demonstration. "
        "Predictions on arbitrary real-world images may be unreliable "
        "because CIFAR-10 contains small, low-resolution images."
    )

st.divider()
st.markdown(
    '<div style="text-align:center;color:#71839F;font-size:12px">'
    'VISIONAI · DEEP LEARNING PROJECT · CNN + CIFAR-10 · PYTORCH'
    '</div>',
    unsafe_allow_html=True,
)
