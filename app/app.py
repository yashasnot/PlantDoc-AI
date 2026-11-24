
import os
import sys
import json

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.segmentation.leaf_segmentation import segment_leaf
from src.segmentation.lesion_segmentation import detect_lesions
from src.explainability.gradcam import generate_gradcam

from src.reliability.gate import (
    assess_prediction_reliability
)

from src.reliability.image_quality import (
    assess_image_quality
)


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_DIR = os.path.join(
    ROOT,
    "models",
    "classification"
)

TEMPERATURE = 0.809876024723053

st.set_page_config(
    page_title="PlantDoc AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f6f8f5;
    color: #17251c;
}

[data-testid="stSidebar"] {
    background: #edf3ed;
    border-right: 1px solid #dce7dc;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #173d2a;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #183b29;
}

.hero {
    background: linear-gradient(
        120deg,
        #123c29,
        #26734a
    );
    color: white;
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 24px;
}

.hero h1 {
    color: white !important;
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    color: #e2f1e5;
    font-size: 15px;
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
    color: #b9e3c5;
}

.metric-card {
    background: white;
    border: 1px solid #e1eae1;
    border-radius: 16px;
    padding: 20px;
    min-height: 112px;
    box-shadow: 0 3px 12px rgba(20, 60, 35, 0.04);
}

.metric-label {
    color: #6d7c70;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #173d2a;
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
    margin-top: 7px;
    overflow-wrap: anywhere;
}

.panel {
    background: white;
    border: 1px solid #e1eae1;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
}

.status-supported {
    background: #e6f5eb;
    color: #17663a;
    border: 1px solid #b9e2c7;
    padding: 16px 18px;
    border-radius: 13px;
    font-weight: 700;
}

.status-provisional {
    background: #fff5df;
    color: #8a5a00;
    border: 1px solid #f0d79d;
    padding: 16px 18px;
    border-radius: 13px;
    font-weight: 700;
}

.status-abstain {
    background: #fcebea;
    color: #a12c2c;
    border: 1px solid #efc1be;
    padding: 16px 18px;
    border-radius: 13px;
    font-weight: 700;
}

.small-muted {
    color: #738176;
    font-size: 12px;
}

div.stButton > button[kind="primary"] {
    background: #26734a;
    border: none;
    border-radius: 11px;
    color: white;
    font-weight: 700;
    padding: 10px 22px;
}

div.stButton > button[kind="primary"]:hover {
    background: #195a37;
    color: white;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1eae1;
    padding: 15px;
    border-radius: 14px;
}

/* ======================================
   SIDEBAR NAVIGATION VISIBILITY FIX
====================================== */

[data-testid="stSidebar"] {
    background: #edf3ed !important;
}

/* All sidebar text */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #244b35 !important;
}

/* Navigation labels */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #244b35 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Navigation label text */
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #244b35 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* Radio button circles */
[data-testid="stSidebar"] [data-testid="stRadio"] label div {
    color: #26734a !important;
}

/* Selected navigation option */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #dcecdf !important;
    border-radius: 9px !important;
}

/* Navigation heading */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #173d2a !important;
    font-weight: 800 !important;
}

/* Sidebar captions */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #718477 !important;
}


</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL
# ==========================================

@st.cache_resource
def load_model():

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    class_path = os.path.join(
        MODEL_DIR,
        "class_names.json"
    )

    checkpoint_path = os.path.join(
        MODEL_DIR,
        "final_classifier.pth"
    )

    with open(class_path, "r") as f:
        classes = json.load(f)

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model, classes, device


model, classes, device = load_model()


# ==========================================
# TRANSFORMS
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# HELPERS
# ==========================================

def metric_card(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_probability(value):
    """
    Display more precision than the old two-decimal
    percentage, and explicitly flag near-saturation.
    """

    percentage = float(value) * 100

    if value >= 1.0:
        return "≥99.9999999%"

    if value >= 0.999999999:
        return f"≥{percentage:.7f}%"

    return f"{percentage:.6f}%"


def clean_class_name(name):

    return (
        name.replace("___", " — ")
        .replace("__", " ")
        .replace("_", " ")
        .replace(",", "")
    )


def status_html(decision):

    if decision == "Supported":

        return """
        <div class="status-supported">
            ✓ SUPPORTED · Strong prediction evidence
        </div>
        """

    if decision == "Provisional":

        return """
        <div class="status-provisional">
            ! PROVISIONAL · Human verification recommended
        </div>
        """

    return """
    <div class="status-abstain">
        × ABSTAIN · Insufficient prediction evidence
    </div>
    """


def run_analysis(image):

    # 1. Reliability
    reliability = assess_prediction_reliability(
        model=model,
        image=image,
        transform=transform,
        device=device,
        temperature=TEMPERATURE
    )

    prediction_index = reliability[
        "prediction_index"
    ]

    disease = classes[prediction_index]

    # 2. Leaf segmentation
    leaf_mask, isolated = segment_leaf(image)

    # 3. Lesion localization
    lesion_mask, overlay, severity = detect_lesions(
        image,
        leaf_mask
    )

    # 4. Image quality
    rgb = np.array(image)

    quality = assess_image_quality(
        rgb,
        leaf_mask
    )

    # 5. Grad-CAM
    tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    cam, _, _ = generate_gradcam(
        model,
        tensor
    )

    heatmap = cv2.applyColorMap(
        np.uint8(
            255 * cv2.resize(
                cam,
                image.size
            )
        ),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    gradcam_overlay = cv2.addWeighted(
        rgb,
        0.55,
        heatmap,
        0.45,
        0
    )

    return {
        "disease": disease,
        "reliability": reliability,
        "leaf_mask": leaf_mask,
        "isolated": isolated,
        "lesion_mask": lesion_mask,
        "overlay": overlay,
        "severity": float(severity),
        "quality": quality,
        "gradcam": gradcam_overlay,
        "original": rgb
    }


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:10px 0 20px 0">
            <div style="font-size:30px">🌿</div>
            <div style="
                font-family:Manrope;
                font-size:23px;
                font-weight:800;
                color:#173d2a;">
                PlantDoc AI
            </div>
            <div style="
                font-size:11px;
                color:#738176;
                letter-spacing:1px;">
                PLANT HEALTH INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Disease Analysis",
            "Visual Evidence",
            "Reliability",
            "Reports"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown(
        "**MODEL INFORMATION**"
    )

    st.caption("EfficientNet-B0 · 38 classes")
    st.caption(f"Device · {device}")
    st.caption(f"Calibration T · {TEMPERATURE:.6f}")

    st.divider()

    st.caption(
        "Research prototype · 2026"
    )


# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            AI-POWERED PLANT HEALTH ANALYSIS
        </div>
        <h1>PlantDoc AI</h1>
        <p>
            Integrated disease recognition, visual evidence,
            severity estimation and prediction reliability.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# OVERVIEW
# ==========================================

if page == "Overview":

    st.subheader("Research Dashboard")

    st.write(
        "An integrated framework for plant disease "
        "classification and evidence-based assessment."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Classification model", "EfficientNet-B0")

    with c2:
        metric_card("Disease categories", "38")

    with c3:
        metric_card("Reliability signals", "2")

    st.markdown("### Analysis workflow")

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            '<div class="panel"><h3>01 · Identify</h3>'
            '<p>Predict the plant disease category.</p></div>',
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            '<div class="panel"><h3>02 · Examine</h3>'
            '<p>Inspect leaf regions, suspected lesions and Grad-CAM.</p></div>',
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            '<div class="panel"><h3>03 · Verify</h3>'
            '<p>Assess confidence and prediction stability.</p></div>',
            unsafe_allow_html=True
        )

    st.info(
        "Open Disease Analysis in the sidebar to upload "
        "a leaf image and begin an assessment."
    )


# ==========================================
# DISEASE ANALYSIS
# ==========================================

elif page == "Disease Analysis":

    st.subheader("Disease Analysis")

    st.write(
        "Upload a clear leaf image for integrated analysis."
    )

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        key="plant_upload"
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded plant leaf",
            width=420
        )

        if st.button(
            "Run PlantDoc Analysis",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Running classification and reliability analysis..."
                ):

                    st.session_state["analysis"] = (
                        run_analysis(image)
                    )

                st.success(
                    "Analysis completed successfully."
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

    result = st.session_state.get("analysis")

    if result:

        reliability = result["reliability"]

        st.markdown("### Assessment result")

        st.markdown(
            status_html(reliability["decision"]),
            unsafe_allow_html=True
        )

        st.markdown("")

        c1, c2 = st.columns(2)

        with c1:
            metric_card(
                "Predicted disease",
                clean_class_name(result["disease"])
            )

        with c2:
            metric_card(
                "Estimated affected area",
                f'{result["severity"]:.2f}%'
            )

        st.markdown("")

        st.markdown(
            "**Reliability explanation**"
        )

        st.write(
            reliability["explanation"]
        )

        st.info(
            "The affected-area value is currently an "
            "HSV-based heuristic estimate, not a validated "
            "agronomic severity measurement."
        )


# ==========================================
# VISUAL EVIDENCE
# ==========================================

elif page == "Visual Evidence":

    st.subheader("Visual Evidence")

    result = st.session_state.get("analysis")

    if not result:

        st.info(
            "Run an image analysis first."
        )

    else:

        st.markdown("### Leaf segmentation")

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                result["leaf_mask"],
                caption="Leaf binary mask",
                clamp=True,
                use_container_width=True
            )

        with c2:
            st.image(
                result["isolated"],
                caption="Isolated leaf",
                use_container_width=True
            )

        st.markdown("### Lesion localization")

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                result["lesion_mask"],
                caption="Suspected lesion mask",
                use_container_width=True
            )

        with c2:
            st.image(
                result["overlay"],
                caption="Lesion overlay",
                use_container_width=True
            )

        st.markdown("### Grad-CAM explanation")

        st.image(
            result["gradcam"],
            caption="Classifier activation heatmap",
            use_container_width=True
        )

        st.caption(
            "Grad-CAM visualizes model activation patterns; "
            "it is not proof of causal disease localization."
        )


# ==========================================
# RELIABILITY
# ==========================================

elif page == "Reliability":

    st.subheader("Prediction Reliability")

    result = st.session_state.get("analysis")

    if not result:

        st.info(
            "Run an image analysis first."
        )

    else:

        r = result["reliability"]

        st.markdown(
            status_html(r["decision"]),
            unsafe_allow_html=True
        )

        st.markdown("")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Raw confidence",
                format_probability(r["raw_confidence"])
            )

        with c2:
            metric_card(
                "Calibrated confidence",
                format_probability(r["calibrated_confidence"])
            )

        with c3:
            metric_card(
                "Variant agreement",
                f'{r["agreement_count"]}/{r["total_variants"]}'
            )

        st.markdown("### Confidence diagnostics")

        st.write(
            f'Raw prediction: '
            f'{clean_class_name(classes[r["raw_prediction_index"]])}'
        )

        st.write(
            f'Calibrated prediction: '
            f'{clean_class_name(classes[r["prediction_index"]])}'
        )

        st.write(
            f'Top-two probability margin: '
            f'{r["probability_margin"]:.10f}'
        )

        st.write(
            f'Temperature: {r["temperature"]:.8f}'
        )

        st.write(
            f'Confidence threshold: '
            f'{r["thresholds"]["confidence"]:.2f}'
        )

        st.write(
            f'Stability threshold: '
            f'{r["thresholds"]["stability"]:.2f}'
        )

        if r["calibrated_confidence"] >= 0.999999:

            st.warning(
                "Near-saturated probability detected. "
                "This is a numerical probability estimate, "
                "not a guarantee of correctness."
            )

        st.markdown("### Top-five predictions")

        top_rows = []

        for index, probability in zip(
            r["top5_indices"],
            r["top5_probabilities"]
        ):

            top_rows.append({
                "Disease": clean_class_name(classes[index]),
                "Calibrated probability": (
                    format_probability(probability)
                )
            })

        st.dataframe(
            top_rows,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Six-variant stability audit")

        variant_rows = []

        for name, prediction in (
            r["variant_predictions"].items()
        ):

            variant_rows.append({
                "Variant": name,
                "Predicted disease": clean_class_name(
                    classes[prediction]
                ),
                "Confidence": format_probability(
                    r["variant_confidences"][name]
                ),
                "Agrees with original": (
                    prediction == r["prediction_index"]
                )
            })

        st.dataframe(
            variant_rows,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Image quality advisory")

        q = result["quality"]

        q1, q2, q3 = st.columns(3)

        with q1:
            metric_card(
                "Brightness",
                f'{q["brightness"]:.2f}'
            )

        with q2:
            metric_card(
                "Contrast",
                f'{q["contrast"]:.2f}'
            )

        with q3:
            metric_card(
                "Blur score",
                f'{q["blur_score"]:.2f}'
            )

        if q["warnings"]:

            st.warning(
                "Image quality warnings: "
                + "; ".join(q["warnings"])
            )

        else:

            st.success(
                "No image quality warnings detected."
            )


# ==========================================
# REPORTS
# ==========================================

elif page == "Reports":

    st.subheader("Analysis Reports")

    result = st.session_state.get("analysis")

    if not result:

        st.info(
            "Complete an image analysis to generate a report."
        )

    else:

        r = result["reliability"]

        report = {
            "project": "PlantDoc AI",

            "classification_model": "EfficientNet-B0",

            "predicted_disease": result["disease"],

            "reliability_decision": r["decision"],

            "raw_confidence": r["raw_confidence"],

            "calibrated_confidence": (
                r["calibrated_confidence"]
            ),

            "prediction_stability": r["stability"],

            "agreement_count": r["agreement_count"],

            "total_variants": r["total_variants"],

            "probability_margin": r["probability_margin"],

            "temperature": r["temperature"],

            "top5_predictions": [
                {
                    "disease": classes[index],
                    "probability": probability
                }
                for index, probability in zip(
                    r["top5_indices"],
                    r["top5_probabilities"]
                )
            ],

            "estimated_affected_area_percent": (
                result["severity"]
            ),

            "severity_method": "HSV heuristic",

            "segmentation_method": "GrabCut",

            "image_quality": result["quality"],

            "reliability_explanation": r["explanation"]
        }

        st.markdown(
            '<div class="panel">'
            '<h3>Analysis report ready</h3>'
            '<p>Download the complete prediction and '
            'reliability diagnostics as JSON.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        st.download_button(
            "Download PlantDoc AI Report",
            data=json.dumps(
                report,
                indent=4
            ),
            file_name="plantdoc_ai_report.json",
            mime="application/json",
            type="primary",
            use_container_width=True
        )

        with st.expander("Preview report data"):

            st.json(report)


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "PlantDoc AI · Research prototype · "
    "Classification, visual evidence and reliability assessment"
)
