"""Streamlit dashboard for medical image diagnosis with Grad-CAM visualization."""

import base64
import io
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Image Diagnosis — ViT",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
CONFIG_OPTIONS = {
    "Chest X-Ray (Pneumonia)": "configs/chest_xray.yaml",
    "Brain Tumor MRI": "configs/brain_tumor.yaml",
    "Skin Cancer": "configs/skin_cancer.yaml",
    "Diabetic Retinopathy": "configs/diabetic_retinopathy.yaml",
}


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(config_path: str, checkpoint_path: str):
    """Load and cache the predictor singleton."""
    try:
        from models import build_model
        from inference.predictor import Predictor

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = cfg.get("dataset", {}).get("num_classes", 2)
        model = build_model(cfg, num_classes=num_classes)
        predictor = Predictor(model=model, cfg=cfg, checkpoint_path=checkpoint_path, device=device)
        return predictor, cfg
    except Exception as e:
        return None, None


def probability_bar_chart(probabilities: dict, class_names: list) -> go.Figure:
    probs = [probabilities.get(c, 0.0) for c in class_names]
    colors = ["#EF4444" if p == max(probs) else "#3B82F6" for p in probs]
    fig = go.Figure(
        go.Bar(
            x=probs,
            y=class_names,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Probability"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(150, 60 * len(class_names)),
    )
    return fig


def image_to_b64(image: np.ndarray) -> str:
    pil = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_metrics_csv(result_dir: str) -> Optional[pd.DataFrame]:
    path = Path(result_dir) / "metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_confusion_matrix_img(result_dir: str) -> Optional[Image.Image]:
    path = Path(result_dir) / "confusion_matrix.png"
    if path.exists():
        return Image.open(path)
    return None


def load_roc_img(result_dir: str) -> Optional[Image.Image]:
    path = Path(result_dir) / "roc_curves.png"
    if path.exists():
        return Image.open(path)
    return None


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/x-ray.png", width=64)
    st.title("Medical Image Diagnosis")
    st.markdown("---")

    task = st.selectbox("Select Task", list(CONFIG_OPTIONS.keys()))
    config_path = CONFIG_OPTIONS[task]

    st.markdown("### Model Checkpoint")
    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=f"checkpoints/{task.lower().replace(' ', '-').replace('(', '').replace(')', '')}/best_model.pth",
    )

    use_tta = st.checkbox("Test-Time Augmentation", value=False)
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    show_attention = st.checkbox("Show Attention Map", value=True)

    st.markdown("---")
    st.markdown("**About**")
    st.caption("ViT-B/16 & EfficientNet | PyTorch | Grad-CAM")

# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
predictor, cfg = None, None
if os.path.exists(checkpoint_path):
    predictor, cfg = load_predictor(config_path, checkpoint_path)
else:
    st.warning(
        f"⚠️ Checkpoint not found at `{checkpoint_path}`. "
        "Train a model first using `scripts/train.py`."
    )

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab_predict, tab_metrics, tab_compare, tab_about = st.tabs(
    ["🔬 Predict", "📊 Metrics", "⚖️ Model Comparison", "ℹ️ About"]
)

# ─── Predict Tab ───────────────────────────────
with tab_predict:
    st.header("Upload Medical Image for Diagnosis")
    uploaded = st.file_uploader(
        "Upload image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="upload"
    )

    if uploaded and predictor:
        col_img, col_result = st.columns([1, 1])

        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        with col_img:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Running inference…"):
            result = predictor.predict(
                image_np,
                generate_gradcam=show_gradcam,
                generate_attention=show_attention,
                use_tta=use_tta,
            )

        with col_result:
            st.markdown("### Prediction")
            confidence_pct = result.confidence * 100
            color = "#22C55E" if confidence_pct > 70 else "#F59E0B" if confidence_pct > 50 else "#EF4444"
            st.markdown(
                f"<h2 style='color:{color}'>{result.predicted_label}</h2>",
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{confidence_pct:.1f}%")

            class_names = cfg.get("dataset", {}).get("classes", list(result.probabilities.keys()))
            fig = probability_bar_chart(result.probabilities, class_names)
            st.plotly_chart(fig, use_container_width=True)

        # Visualizations
        if result.gradcam_overlay is not None or result.attention_overlay is not None:
            st.markdown("---")
            st.subheader("Explainability Visualizations")
            vis_cols = [c for c in [
                ("Original", image_np),
                ("Grad-CAM Overlay", result.gradcam_overlay),
                ("Attention Map", result.attention_overlay),
            ] if c[1] is not None]

            cols = st.columns(len(vis_cols))
            for col, (title, img) in zip(cols, vis_cols):
                col.image(img, caption=title, use_column_width=True)

    elif uploaded and not predictor:
        st.error("Model not loaded. Please check checkpoint path in the sidebar.")

# ─── Metrics Tab ───────────────────────────────
with tab_metrics:
    st.header("Evaluation Metrics")
    if cfg:
        result_dir = cfg.get("project", {}).get("output_dir", "results")
        metrics_df = load_metrics_csv(result_dir)
        if metrics_df is not None:
            st.dataframe(metrics_df.T.rename(columns={0: "Value"}), use_container_width=True)
            # Key metrics radar
            key_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            vals = [metrics_df.get(m, pd.Series([0])).values[0] for m in key_metrics]
            fig = go.Figure(
                go.Scatterpolar(r=vals, theta=key_metrics, fill="toself", name="Model")
            )
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        cm_img = load_confusion_matrix_img(result_dir)
        roc_img = load_roc_img(result_dir)
        if cm_img:
            col1.image(cm_img, caption="Confusion Matrix", use_column_width=True)
        if roc_img:
            col2.image(roc_img, caption="ROC Curves", use_column_width=True)
        if not metrics_df and not cm_img:
            st.info("No evaluation results yet. Run `scripts/evaluate.py` first.")
    else:
        st.info("Select a task and load a model to view metrics.")

# ─── Model Comparison Tab ──────────────────────
with tab_compare:
    st.header("Model Architecture Comparison")
    comparison_path = "results/model_comparison.csv"
    if os.path.exists(comparison_path):
        df = pd.read_csv(comparison_path, index_col="model")
        st.dataframe(df.style.highlight_max(axis=0, color="#BBF7D0"), use_container_width=True)

        # Bar chart comparison
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        metric_choice = st.selectbox("Metric to compare", numeric_cols)
        fig = px.bar(df.reset_index(), x="model", y=metric_choice, color="model", text=metric_choice)
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No comparison data. Run `scripts/compare_models.py` first.")

# ─── About Tab ─────────────────────────────────
with tab_about:
    st.header("About This Project")
    st.markdown("""
    ## Medical Image Diagnosis System — Vision Transformers

    A **production-quality AI system** for classifying medical images across four tasks:

    | Task | Classes | Dataset |
    |------|---------|---------|
    | Chest X-Ray | Normal, Pneumonia | Kaggle CXR |
    | Brain Tumor MRI | Glioma, Meningioma, Pituitary, No Tumor | Kaggle Brain MRI |
    | Skin Cancer | Benign, Malignant | HAM10000 |
    | Diabetic Retinopathy | 0–4 severity | APTOS 2019 |

    ### Architecture
    - **ViT-B/16** — Vision Transformer (patch size 16, 224×224)
    - **EfficientNet-B0 / B3** — CNN baseline comparison

    ### Explainability
    - **Grad-CAM++** for CNN models
    - **Attention Rollout** for ViT models

    ### Tech Stack
    `PyTorch` · `timm` · `Albumentations` · `FastAPI` · `Streamlit` · `Docker`

    ---
    Built as a professional ML engineering portfolio project.
    """)
