# =========================================================
# app.py — MAIN STREAMLIT APPLICATION (FINAL)
# =========================================================

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

import torch
import timm
import xgboost as xgb
from PIL import Image
from torchvision import transforms

from auth import login, register
from database import (
    create_user, get_user_by_email, get_all_users, delete_user,
    save_report, get_patient_reports, get_all_reports, add_doctor_remark
)
from pdf_report import generate_pdf

# =========================================================
# Streamlit Config
# =========================================================
st.set_page_config(
    page_title="Lung Cancer Detection System",
    page_icon="🫁",
    layout="centered"
)

# =========================================================
# GLOBAL CSS (overlay + buttons)
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
# Session State
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "index"

if "user" not in st.session_state:
    st.session_state.user = None

def go(page):
    st.session_state.page = page
    st.rerun()

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
# Load Landing Page (HTML + CSS)
# =========================================================
def load_index_page():
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    css = Path("frontend/style.css").read_text(encoding="utf-8")

    components.html(
        f"<style>{css}</style>{html}",
        height=600
    )

# =========================================================
# INDEX PAGE
# =========================================================
if st.session_state.page == "index":
    load_index_page()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        if st.button("🔐 Login", use_container_width=True):
            go("login")

        if st.button("📝 Register", use_container_width=True):
            go("register")

# =========================================================
# LOGIN PAGE
# =========================================================
elif st.session_state.page == "login":
    st.markdown("## 🔐 Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("🔓 Login", use_container_width=True):
        user = login(email, password)
        if user:
            st.session_state.user = {
                "id": user[0],
                "name": user[1],
                "email": user[2],
                "role": user[4]
            }
            go("dashboard")
        else:
            st.error("Invalid email or password")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 New User? Register", use_container_width=True):
            go("register")

    with col2:
        if st.button("⬅ Back to Home", use_container_width=True):
            go("index")

# =========================================================
# REGISTER PAGE
# =========================================================
elif st.session_state.page == "register":
    st.markdown("## 📝 New User Registration")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Register As", ["patient", "doctor"])

    if st.button("✅ Register", use_container_width=True):
        if get_user_by_email(email):
            st.error("User already exists")
        else:
            register(name, email, password, role)
            st.success("Registration successful! Please login.")
            go("login")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔐 Already Registered? Login", use_container_width=True):
            go("login")

    with col2:
        if st.button("⬅ Back to Home", use_container_width=True):
            go("index")

# =========================================================
# DASHBOARD (ROLE-BASED)
# =========================================================
elif st.session_state.page == "dashboard":
    user = st.session_state.user

    st.sidebar.success(f"Logged in as {user['role']}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        go("index")

    # =====================================================
    # PATIENT DASHBOARD
    # =====================================================
    if user["role"] == "patient":
        st.markdown("## 🧑‍🦱 Patient Dashboard")
        st.info("Upload CT image • Predict • Download PDF")

        uploaded = st.file_uploader(
            "Upload Lung CT Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_column_width=True)

            if st.button("🔍 Predict"):
                img_tensor = transform(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    features = swin_model(img_tensor).cpu().numpy()

                prob = xgb_model.predict_proba(features)[0]
                benign, malignant = float(prob[0]), float(prob[1])
                prediction = "Malignant" if malignant > 0.5 else "Benign"

                st.success(f"🧠 Prediction: {prediction}")
                st.progress(malignant)

                pdf_path = generate_pdf(
                    user["name"], prediction, benign, malignant
                )

                save_report(
                    user["id"], prediction, benign, malignant, pdf_path
                )

                st.download_button(
                    "📄 Download Medical Report",
                    open(pdf_path, "rb"),
                    file_name="lung_report.pdf"
                )

        st.divider()
        st.subheader("📜 Your Medical History")

        reports = get_patient_reports(user["id"])
        for r in reports:
            st.markdown(f"""
            **Report ID:** {r[0]}  
            **Prediction:** {r[2]}  
            **Benign:** {r[3]:.2f} | **Malignant:** {r[4]:.2f}  
            **Doctor Remark:** {r[6] or "Pending"}
            """)
            st.download_button(
                "Download PDF",
                open(r[5], "rb"),
                file_name="report.pdf",
                key=f"p_{r[0]}"
            )
            st.divider()

    # =====================================================
    # DOCTOR DASHBOARD
    # =====================================================
    elif user["role"] == "doctor":
        st.markdown("## 👨‍⚕️ Doctor Dashboard")
        st.info("Review reports & add remarks")

        reports = get_all_reports()
        for r in reports:
            st.markdown(f"""
            **Report ID:** {r[0]}  
            **Patient ID:** {r[1]}  
            **Prediction:** {r[2]}  
            **Benign:** {r[3]:.2f} | **Malignant:** {r[4]:.2f}
            """)

            remark = st.text_area(
                "Doctor Remark",
                value=r[6] or "",
                key=f"remark_{r[0]}"
            )

            if st.button("Save Remark", key=f"save_{r[0]}"):
                add_doctor_remark(r[0], remark)
                st.success("Remark saved")
                st.rerun()

            st.divider()

    # =====================================================
    # ADMIN DASHBOARD
    # =====================================================
    elif user["role"] == "admin":
        st.markdown("## 🛠️ Admin Dashboard")
        st.info("User management")

        st.subheader("➕ Add New User")
        a_name = st.text_input("Name")
        a_email = st.text_input("Email")
        a_password = st.text_input("Password", type="password")
        a_role = st.selectbox("Role", ["patient", "doctor", "admin"])

        if st.button("Create User"):
            if get_user_by_email(a_email):
                st.error("User already exists")
            else:
                create_user(a_name, a_email, a_password, a_role)
                st.success("User created")
                st.rerun()

        st.divider()
        st.subheader("👥 All Users")

        users = get_all_users()
        for u in users:
            st.markdown(
                f"**ID:** {u[0]} | **Name:** {u[1]} | **Email:** {u[2]} | **Role:** {u[3]}"
            )
            if st.button("❌ Delete User", key=f"del_{u[0]}"):
                delete_user(u[0])
                st.warning("User deleted")
                st.rerun()