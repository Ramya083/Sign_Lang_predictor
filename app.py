"""
app.py — ASL Sign Language Translator
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="ASL Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.main-header h1 { color: #e94560; font-size: 2.2rem; font-weight: 700; margin: 0 0 0.3rem; }
.main-header p  { color: rgba(255,255,255,0.65); margin: 0; font-size: 1rem; }

.pred-card {
    background: #1a1a2e;
    border: 1px solid rgba(233,69,96,0.3);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.pred-letter {
    font-size: 5rem; font-weight: 700; color: #e94560;
    line-height: 1; font-family: 'JetBrains Mono', monospace;
}
.pred-conf { font-size: 0.85rem; color: rgba(255,255,255,0.5); margin-top: 0.4rem; }

.sentence-box {
    background: #16213e;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 1.5rem; font-weight: 600; color: #ffffff;
    min-height: 64px; letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace; word-break: break-all;
}
.top3-row { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 0.5rem; }
.top3-chip {
    background: rgba(233,69,96,0.15); border: 1px solid rgba(233,69,96,0.3);
    border-radius: 20px; padding: 3px 12px; font-size: 0.8rem; color: #e94560;
    font-family: 'JetBrains Mono', monospace;
}
.status-ok   { color: #4ade80; font-weight: 600; }
.status-err  { color: #f87171; font-weight: 600; }
.tip-box {
    background: rgba(255,255,255,0.04); border-left: 3px solid #e94560;
    border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
    font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 0.6rem;
}
.stButton > button {
    background: #e94560 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; width: 100% !important;
}
section[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: 1px solid rgba(255,255,255,0.07);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    from utils.predictor import ASLPredictor
    return ASLPredictor("model/asl_classifier.pkl")

@st.cache_resource
def load_hand_detector():
    from utils.hand_utils import get_hands_detector
    return get_hands_detector(static_mode=True)


model_ready = os.path.exists("model/asl_classifier.pkl")

st.markdown("""
<div class="main-header">
  <h1>🤟 ASL Sign Language Translator</h1>
  <p>Upload a hand image or use your webcam → get real-time ASL letter translation</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    confidence_threshold = st.slider("Min confidence threshold", 0.1, 0.99, 0.40, 0.05)

    st.markdown("---")
    st.markdown("## 📖 Supported Signs")
    st.markdown("Letters: **A–Y** (excl. J, Z)")
    st.markdown("Special: **space**, **del**, **nothing**")

    st.markdown("---")
    st.markdown("## 💡 Tips")
    for t in ["Good lighting", "Plain background", "Hand centered", "Fingers fully visible", "Hold sign steady"]:
        st.markdown(f'<div class="tip-box">→ {t}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🚀 Share via ngrok")
    ngrok_token = st.text_input("ngrok authtoken", type="password")
    if st.button("Start ngrok tunnel"):
        if ngrok_token:
            try:
                from pyngrok import ngrok, conf
                conf.get_default().auth_token = ngrok_token
                tunnel = ngrok.connect(8501)
                st.success(f"Public URL:\n{tunnel.public_url}")
                st.session_state["ngrok_url"] = tunnel.public_url
            except Exception as e:
                st.error(f"ngrok error: {e}")
        else:
            st.warning("Enter your ngrok authtoken first.")
    if "ngrok_url" in st.session_state:
        st.info(f"🌐 Live at:\n{st.session_state['ngrok_url']}")


if not model_ready:
    st.error("⚠️ Model not found. Run:  `python train_model.py`  then restart the app.")
    st.stop()

predictor = load_predictor()
hands     = load_hand_detector()

if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "history" not in st.session_state:
    st.session_state.history = []

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("### 📷 Input")
    input_mode = st.radio("Input mode", ["Upload Image", "Webcam Snapshot"], horizontal=True, label_visibility="collapsed")

    uploaded_image = None

    if input_mode == "Upload Image":
        f = st.file_uploader("Upload hand sign image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
        if f:
            uploaded_image = np.array(Image.open(f).convert("RGB"))
    else:
        cam = st.camera_input("Take a photo of your hand sign")
        if cam:
            uploaded_image = np.array(Image.open(cam).convert("RGB"))

    if uploaded_image is not None:
        from utils.hand_utils import extract_landmarks, normalize_landmarks

        landmarks, annotated, detected = extract_landmarks(uploaded_image, hands)
        st.image(annotated, caption="Detected hand landmarks" if detected else "No hand detected", use_container_width=True)

        if not detected:
            st.markdown('<p class="status-err">✗ No hand detected. Try better lighting or a plain background.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-ok">✓ Hand detected successfully</p>', unsafe_allow_html=True)
            norm_lm = normalize_landmarks(landmarks)
            label, confidence, top3 = predictor.predict(norm_lm)

            with col_right:
                st.markdown("### 🔤 Prediction")
                conf_pct     = round(confidence * 100, 1)
                conf_color   = "#4ade80" if confidence >= confidence_threshold else "#fbbf24"
                display_label = label if confidence >= confidence_threshold else "?"

                st.markdown(f"""
                <div class="pred-card">
                  <div class="pred-letter">{display_label}</div>
                  <div class="pred-conf" style="color:{conf_color}">Confidence: {conf_pct}%</div>
                  <div class="top3-row">
                    {"".join(f'<span class="top3-chip">{l}: {p}%</span>' for l,p in top3)}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(confidence, text=f"{conf_pct}% confident")

                st.markdown("---")
                st.markdown("### ✍️ Build Sentence")
                b1, b2, b3 = st.columns(3)

                with b1:
                    if st.button(f"Add '{display_label}'", disabled=(display_label == "?")):
                        if display_label == "space":
                            st.session_state.sentence += " "
                        elif display_label == "del":
                            st.session_state.sentence = st.session_state.sentence[:-1]
                        elif display_label != "nothing":
                            st.session_state.sentence += display_label
                        st.session_state.history.append(display_label)
                with b2:
                    if st.button("⌫ Backspace"):
                        st.session_state.sentence = st.session_state.sentence[:-1]
                with b3:
                    if st.button("🗑 Clear"):
                        st.session_state.sentence = ""
                        st.session_state.history = []

                st.markdown("---")
                st.markdown("### 💬 Sentence")
                st.markdown(f'<div class="sentence-box">{st.session_state.sentence or "..."}</div>', unsafe_allow_html=True)

                if st.session_state.history:
                    st.markdown(f"**History:** `{'` → `'.join(st.session_state.history[-10:])}`")

    else:
        with col_right:
            st.markdown("### 🔤 Prediction")
            st.markdown('<div class="pred-card"><div class="pred-letter" style="opacity:0.3">—</div><div class="pred-conf">Waiting for input…</div></div>', unsafe_allow_html=True)
            st.markdown("### 💬 Sentence")
            st.markdown(f'<div class="sentence-box">{st.session_state.sentence or "..."}</div>', unsafe_allow_html=True)


st.markdown("---")
st.markdown("### 🗂 ASL Alphabet Reference")
labels = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ["del", "nothing", "space"]
cols   = st.columns(9)
for i, letter in enumerate(labels):
    with cols[i % 9]:
        st.markdown(
            f'<div style="background:#1a1a2e;border:1px solid rgba(255,255,255,0.1);border-radius:8px;'
            f'padding:8px 4px;text-align:center;color:white;font-weight:600;font-size:0.9rem;'
            f'margin-bottom:6px;font-family:monospace;">{letter}</div>',
            unsafe_allow_html=True
        )

st.markdown("---")
st.markdown('<div style="text-align:center;color:rgba(255,255,255,0.3);font-size:0.8rem;padding:1rem 0;">Built with MediaPipe · scikit-learn · Streamlit</div>', unsafe_allow_html=True)