# =========================================================
# app.py — LUNG CANCER IMAGE ANALYSIS (REVIEWER VERSION)
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
# GLOBAL CSS
# =========================================================
st.markdown("""
<style>
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 14px;
    font-size: 18px;
    height: 56px;
    font-weight: bold;
    border: none;
}
.stButton > button:hover {
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown("## 🫁 Lung Cancer Detection from CT Images")
st.info("Upload a lung CT scan image to analyze whether it is **Benign or Malignant**.")

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
        with st.spinner("Analyzing image..."):
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                features = swin_model(img_tensor).cpu().numpy()

            prob = xgb_model.predict_proba(features)[0]
            benign, malignant = float(prob[0]), float(prob[1])

            prediction = "Malignant" if malignant > 0.5 else "Benign"

        st.success(f"🧠 Prediction Result: **{prediction}**")

        st.markdown("### 📊 Confidence Scores")
        st.progress(malignant)

        col1, col2 = st.columns(2)
        col1.metric("Benign Probability", f"{benign:.2f}")
        col2.metric("Malignant Probability", f"{malignant:.2f}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "⚠️ This system is developed for academic and research demonstration purposes only."
)