# =========================================================
# app.py — LUNG CANCER IMAGE ANALYSIS (REVIEWER READY)
# =========================================================

import streamlit as st
import torch
import timm
import xgboost as xgb
from PIL import Image
from torchvision import transforms

# =========================================================
# Streamlit Config
# =========================================================
st.set_page_config(
    page_title="Lung Cancer Detection System",
    page_icon="🫁",
    layout="centered"
)

# =========================================================
# GLOBAL CSS (Professional + Animations)
# =========================================================
st.markdown("""
<style>
/* Button Styling */
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 14px;
    font-size: 18px;
    height: 56px;
    font-weight: bold;
    border: none;
    transition: transform 0.2s ease;
}
.stButton > button:hover {
    transform: scale(1.03);
}

/* Fade & Slide Animation */
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeSlide 0.6s ease-out;
}

/* Result Card */
.result-card {
    background: #0f172a;
    padding: 22px;
    border-radius: 16px;
    margin-top: 20px;
    border: 1px solid #1e293b;
}

/* Image Hover */
img {
    border-radius: 14px;
    transition: transform 0.3s ease;
}
img:hover {
    transform: scale(1.01);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("## 🫁 Lung Cancer Detection from CT Images")
st.info(
    "Upload a lung CT scan image to analyze whether the case is "
    "**Benign or Malignant**"
)

# =========================================================
# Device
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Load Models
# =========================================================
@st.cache_resource
def load_models():
    swin = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=0
    ).to(DEVICE).eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("lung_cancer_xgb.json")

    return swin, xgb_model

swin_model, xgb_model = load_models()

# =========================================================
# Image Transform
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================================================
# IMAGE UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "📤 Upload Lung CT Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded CT Image", use_container_width=True)

    if st.button("🔍 Analyze Image", use_container_width=True):
        status = st.empty()

        status.markdown(
            "<div class='fade-in'>🧠 <b>Analyzing CT image...</b></div>",
            unsafe_allow_html=True
        )

        with st.spinner("Extracting deep features & running XGBoost classifier..."):
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                features = swin_model(img_tensor).cpu().numpy()

            prob = xgb_model.predict_proba(features)[0]
            benign, malignant = float(prob[0]), float(prob[1])
            prediction = "Malignant" if malignant > 0.5 else "Benign"

        # =================================================
        # RESULT DISPLAY
        # =================================================
        st.markdown(f"""
        <div class="result-card fade-in">
            <h3>🧠 Prediction Result</h3>
            <h2 style="color:#38bdf8;">{prediction}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='fade-in'><h4>📊 Confidence Scores</h4></div>",
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Benign Probability", f"{benign:.2f}")
        with col2:
            st.metric("Malignant Probability", f"{malignant:.2f}")

        st.progress(malignant)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
