import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import time
import plotly.graph_objects as go

# Memperbaiki error deserialization GlorotUniform akibat ketidakcocokan versi Keras
original_glorot_init = keras.initializers.GlorotUniform.__init__

def safe_glorot_init(self, seed=None, **kwargs):
    kwargs.pop('input_axes', None)
    kwargs.pop('output_axes', None)
    original_glorot_init(self, seed=seed, **kwargs)

keras.initializers.GlorotUniform.__init__ = safe_glorot_init

original_zeros_init = keras.initializers.Zeros.__init__
def safe_zeros_init(self, **kwargs):
    kwargs.pop('input_axes', None)
    kwargs.pop('output_axes', None)
    original_zeros_init(self, **kwargs)

keras.initializers.Zeros.__init__ = safe_zeros_init

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Penyakit Ayam",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Modern CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Card Styling (Auto adapts to Light/Dark mode) */
.metric-card {
    background-color: rgba(128, 128, 128, 0.05);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.metric-title {
    font-size: 0.875rem;
    font-weight: 600;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.metric-sub {
    font-size: 0.875rem;
    opacity: 0.6;
}

/* Badges */
.badge-healthy { background-color: rgba(34, 197, 94, 0.2); color: #4ade80; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(34, 197, 94, 0.3); }
.badge-sick { background-color: rgba(239, 68, 68, 0.2); color: #f87171; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(239, 68, 68, 0.3); }
.badge-warning { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(245, 158, 11, 0.3); }

/* Typography */
h1, h2, h3 {
    font-weight: 700;
}

hr {
    border-color: rgba(128, 128, 128, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BASELINE_PATH = os.path.join(BASE_DIR, "model_outputs", "MobileNetV3_Baseline", "model_baseline_final.keras")
MODEL_DCGAN_PATH    = os.path.join(BASE_DIR, "model_outputs", "MobileNetV3_DCGAN", "model_dcgan_final.keras")

CLASS_NAMES = ["Cocci", "Healthy", "NCD", "Salmo"]
CLASS_INFO = {
    "Cocci": {
        "status": "sick",
        "label": "Koksidiosis",
        "desc": "Koksidiosis (Coccidiosis) adalah penyakit parasit usus yang disebabkan oleh protozoa Eimeria.",
        "action": "Segera berikan obat anti-koksidia dan isolasi ayam yang terinfeksi untuk mencegah penularan."
    },
    "Healthy": {
        "status": "healthy",
        "label": "Sehat",
        "desc": "Ayam dalam kondisi sehat dan tidak menunjukkan gejala klinis penyakit dari kotorannya.",
        "action": "Lanjutkan standar perawatan rutin, jaga sanitasi kandang, dan lakukan pemantauan berkala."
    },
    "NCD": {
        "status": "sick",
        "label": "Newcastle Disease",
        "desc": "Newcastle Disease (Tetelo) adalah penyakit virus menular yang menyerang sistem pernapasan dan saraf.",
        "action": "Lakukan vaksinasi darurat (jika memungkinkan), karantina area ketat, dan desinfeksi total kandang."
    },
    "Salmo": {
        "status": "warning",
        "label": "Salmonellosis",
        "desc": "Salmonellosis adalah infeksi bakteri Salmonella yang dapat menyebar melalui makanan atau air yang terkontaminasi.",
        "action": "Berikan antibiotik sesuai anjuran dokter hewan, pastikan pakan/minum bersih, dan perbaiki sirkulasi udara."
    },
}

IMG_SIZE = (224, 224)

# ─── Logic ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, pil_img: Image.Image):
    x = preprocess_image(pil_img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]), preds

def render_chart(preds: np.ndarray, title: str = "Confidence Distribution"):
    percentages = [round(float(p) * 100, 2) for p in preds]
    # Use softer colors for dark mode compatibility
    max_idx = np.argmax(percentages)
    marker_colors = ['rgba(128, 128, 128, 0.3)'] * 4
    marker_colors[max_idx] = '#3b82f6' # Blue highlight for predicted class

    fig = go.Figure(go.Bar(
        x=CLASS_NAMES,
        y=percentages,
        text=[f"{v:.1f}%" for v in percentages],
        textposition="outside",
        textfont=dict(size=12),
        marker_color=marker_colors,
        marker_line_color=marker_colors,
        marker_line_width=1,
        opacity=0.9
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, max(percentages) + 15], showgrid=True, gridcolor="rgba(128, 128, 128, 0.1)", ticksuffix="%"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2600/2600028.png", width=60) # Simple icon
    st.markdown("## PoultryVision")
    st.markdown("<p style='opacity:0.7; font-size:0.9rem;'>Sistem Klasifikasi Cerdas Penyakit Ayam menggunakan Deep Learning.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### Spesifikasi Model")
    st.markdown("""
    <div style='background-color:rgba(128,128,128,0.05); padding:1rem; border-radius:8px; border:1px solid rgba(128,128,128,0.2); font-size:0.85rem;'>
        <b>Arsitektur:</b> MobileNetV3-Small<br>
        <b>Input Size:</b> 224x224 px<br>
        <b>Output:</b> 4 Kelas Penyakit<br>
        <b>Format:</b> .keras
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; opacity:0.5; font-size:0.75rem;'>© 2026 UNIKOM</p>", unsafe_allow_html=True)

# ─── Main Interface ────────────────────────────────────────────────────────────
st.title("Dashboard Deteksi Penyakit Ayam")
st.markdown("<p style='opacity:0.8; font-size:1.1rem; margin-bottom:2rem;'>Analisis kondisi ayam berdasarkan citra kotoran dengan model MobileNetV3 yang ditingkatkan (DCGAN).</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Model Baseline", "✨ Model DCGAN (Balanced)", "⚖️ Perbandingan Model"])

def render_result(pred_class, pred_conf, infer_time, arr):
    info = CLASS_INFO[pred_class]
    badge_class = f"badge-{info['status']}"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Hasil Analisis AI</div>
        <div class="metric-value">{pred_class} <span class="{badge_class}" style="vertical-align:middle; margin-left:8px;">{info['label']}</span></div>
        <div class="metric-sub">Confidence: <b>{pred_conf*100:.2f}%</b> &nbsp;•&nbsp; Waktu Inferensi: <b>{infer_time*1000:.1f} ms</b></div>
        <hr style="margin:1rem 0;">
        <div style="font-size:0.9rem; opacity:0.8; margin-bottom:0.5rem;">
            <b>Deskripsi:</b><br>{info['desc']}
        </div>
        <div style="font-size:0.9rem; opacity:0.8;">
            <b>Rekomendasi Tindakan:</b><br>{info['action']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(render_chart(arr, "Probabilitas Kelas"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — BASELINE
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col_input, col_output = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.markdown("#### 1. Unggah Gambar Input")
        file_base = st.file_uploader("Format didukung: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"], key="file_base")
        if file_base:
            img_base = Image.open(file_base)
            # Dihapus parameter style="" yang menyebabkan error
            st.image(img_base, use_column_width=True)
            
            btn_base = st.button("Mulai Analisis (Baseline)", type="primary", use_container_width=True)
        else:
            btn_base = False

    with col_output:
        st.markdown("#### 2. Hasil Deteksi")
        if file_base and btn_base:
            with st.spinner("Memuat model dan menganalisis..."):
                try:
                    model_b = load_keras_model(MODEL_BASELINE_PATH)
                    start = time.time()
                    pred_cls, pred_cnf, pred_arr = predict(model_b, img_base)
                    infer = time.time() - start
                    render_result(pred_cls, pred_cnf, infer, pred_arr)
                except Exception as e:
                    st.error(f"Gagal memuat model: {str(e)}")
        elif not file_base:
            st.info("👈 Silakan unggah gambar di panel kiri terlebih dahulu.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — DCGAN
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    col_input, col_output = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.markdown("#### 1. Unggah Gambar Input")
        file_dcgan = st.file_uploader("Format didukung: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"], key="file_dcgan")
        if file_dcgan:
            img_dcgan = Image.open(file_dcgan)
            # Dihapus parameter style="" yang menyebabkan error
            st.image(img_dcgan, use_column_width=True)
            
            btn_dcgan = st.button("Mulai Analisis (DCGAN)", type="primary", use_container_width=True)
        else:
            btn_dcgan = False

    with col_output:
        st.markdown("#### 2. Hasil Deteksi")
        if file_dcgan and btn_dcgan:
            with st.spinner("Memuat model dan menganalisis..."):
                try:
                    model_d = load_keras_model(MODEL_DCGAN_PATH)
                    start = time.time()
                    pred_cls, pred_cnf, pred_arr = predict(model_d, img_dcgan)
                    infer = time.time() - start
                    render_result(pred_cls, pred_cnf, infer, pred_arr)
                except Exception as e:
                    st.error(f"Gagal memuat model: {str(e)}")
        elif not file_dcgan:
            st.info("👈 Silakan unggah gambar di panel kiri terlebih dahulu.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("Bandingkan performa prediksi antara model **Baseline** (imbalanced) dengan model **DCGAN** (balanced).")
    
    col_img, col_res = st.columns([1, 2], gap="large")
    
    with col_img:
        file_comp = st.file_uploader("Pilih gambar pengujian", type=["jpg", "jpeg", "png"], key="file_comp")
        if file_comp:
            img_comp = Image.open(file_comp)
            # Dihapus parameter style="" yang menyebabkan error
            st.image(img_comp, use_column_width=True)
            btn_comp = st.button("Bandingkan Sekarang", type="primary", use_container_width=True)
        else:
            btn_comp = False
            
    with col_res:
        if file_comp and btn_comp:
            with st.spinner("Menjalankan kedua model..."):
                try:
                    mb = load_keras_model(MODEL_BASELINE_PATH)
                    md = load_keras_model(MODEL_DCGAN_PATH)
                    
                    pb_cls, pb_cnf, pb_arr = predict(mb, img_comp)
                    pd_cls, pd_cnf, pd_arr = predict(md, img_comp)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-card" style="border-top: 4px solid #94a3b8;">
                            <div class="metric-title">Model Baseline</div>
                            <div class="metric-value" style="font-size:1.5rem;">{pb_cls}</div>
                            <div class="metric-sub">Confidence: <b>{pb_cnf*100:.1f}%</b></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card" style="border-top: 4px solid #3b82f6;">
                            <div class="metric-title">Model DCGAN</div>
                            <div class="metric-value" style="font-size:1.5rem;">{pd_cls}</div>
                            <div class="metric-sub">Confidence: <b>{pd_cnf*100:.1f}%</b></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Grouped Bar Chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=CLASS_NAMES, y=[round(float(p)*100, 2) for p in pb_arr],
                        name='Baseline', marker_color='rgba(128, 128, 128, 0.4)'
                    ))
                    fig.add_trace(go.Bar(
                        x=CLASS_NAMES, y=[round(float(p)*100, 2) for p in pd_arr],
                        name='DCGAN', marker_color='#3b82f6'
                    ))
                    fig.update_layout(
                        barmode='group',
                        title=dict(text="Perbandingan Probabilitas Tiap Kelas", font=dict(size=14)),
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20),
                        yaxis=dict(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor="rgba(128, 128, 128, 0.1)"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Gagal memuat model: {str(e)}")
        elif not file_comp:
            st.info("Unggah gambar dan klik Bandingkan Sekarang untuk melihat analisis komparatif.")
