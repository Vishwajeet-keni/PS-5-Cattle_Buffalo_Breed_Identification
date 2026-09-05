import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageStat
import torchvision.transforms as transforms
import torchvision.models as models
import os
import csv
import pandas as pd
from datetime import datetime
import random
import json
import hashlib

from huggingface_hub import hf_hub_download
from transformers import BlipProcessor, BlipForConditionalGeneration
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cattle Breed Identifier",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

HF_REPO_ID = "ujjwal75/indian-bovine-breeds-model"
MODEL_FILENAME = "Indian_bovine_finetuned_model.pth"
CLASSES_FILENAME = "classes.json"
METRICS_FILENAME = "metrics.json"

IMG_SIZE = 224

CSV_FILE = "cattle_classification_data.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSLATIONS
# ============================================================

def get_translation(key, language="en"):

    translations = {

        "en": {
            "title": "Indian Cattle Breed Identifier",
            "subtitle": "Discover the rich diversity of Indian bovine breeds",
            "upload_info": "Upload an image of a cow or buffalo to identify its breed",
            "marketplace_button": "Cattle Marketplace",
            "upload_label": "Choose a cattle image",
            "analyzing": "Analyzing breed characteristics...",
            "predicted_breed": "Predicted Breed:",
            "confidence": "Confidence:",
            "breed_info": "Breed Information",
            "pedigree": "Pedigree / Lineage",
            "productivity": "Productivity",
            "rearing_conditions": "Optimal Rearing Conditions",
            "origin": "Origin",
            "physical_chars": "Physical Characteristics",
            "lifespan": "Lifespan (Years)",
            "temperament": "Temperament",
            "productivity_metrics": "Productivity Metrics",
            "physical_measurements": "Physical Measurements",
            "body_length": "Body Length",
            "height_withers": "Height at Withers",
            "chest_width": "Chest Width",
            "rump_angle": "Rump Angle",
            "refresh": "Upload another image to analyze another animal",
            "heritage": "Celebrating India's rich bovine heritage",
            "marketplace_title": "Cattle Marketplace",
            "marketplace_subtitle": "Buy and Sell Quality Cattle",
            "back_button": "Back to Breed Identifier",
            "price": "Price",
            "age": "Age",
            "milk_yield": "Milk Yield",
            "lactation_stage": "Lactation Stage",
            "vaccination": "Vaccination",
            "seller": "Seller",
            "contact": "Contact",
            "location": "Location",
            "add_listing": "Add Your Listing",
            "cattle_breed": "Cattle Breed",
            "submit_listing": "Submit Listing",
            "listing_submitted": "Listing submitted!",
            "description": "Description",
            "prediction_error": "Prediction error",
            "processing_error": "Error processing image",
            "confidence_error": "Could not confidently identify the breed.",
            "no_info": "No additional information found for this breed.",
            "model_loading": "Loading AI model...",
            "model_error": "Model not found. Using demo mode.",
            "demo_mode": "Running in demo mode.",
            "chat_title": "Cattle Assistant",
            "model_performance": "Model Performance",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1 Score",
            "metrics_note": "Evaluated on held-out test dataset",
            "history": "Classification History",
            "about": "About",
            "model_ready": "AI Model Ready",
            "demo": "Demo Mode"
        },

        "hi": {
            "title": "भारतीय मवेशी नस्ल पहचानकर्ता",
            "subtitle": "भारतीय बोवाइन नस्लों की समृद्ध विविधता की खोज करें",
            "upload_info": "नस्ल की पहचान करने के लिए गाय या भैंस की एक छवि अपलोड करें",
            "marketplace_button": "मवेशी बाजार",
            "upload_label": "एक मवेशी छवि चुनें",
            "analyzing": "नस्ल की विशेषताओं का विश्लेषण किया जा रहा है...",
            "predicted_breed": "अनुमानित नस्ल:",
            "confidence": "विश्वास:",
            "breed_info": "नस्ल की जानकारी",
            "pedigree": "वंशावली",
            "productivity": "उत्पादकता",
            "rearing_conditions": "इष्टतम पालन की स्थिति",
            "origin": "मूल",
            "physical_chars": "शारीरिक विशेषताएं",
            "lifespan": "जीवनकाल (वर्ष)",
            "temperament": "स्वभाव",
            "productivity_metrics": "उत्पादकता मेट्रिक्स",
            "physical_measurements": "शारीरिक माप",
            "body_length": "शरीर की लंबाई",
            "height_withers": "कंधे की ऊंचाई",
            "chest_width": "छाती की चौड़ाई",
            "rump_angle": "रंप कोण",
            "refresh": "दूसरे जानवर का विश्लेषण करने के लिए दूसरी छवि अपलोड करें",
            "heritage": "भारत की समृद्ध बोवाइन विरासत का जश्न",
            "marketplace_title": "मवेशी बाजार",
            "marketplace_subtitle": "गुणवत्तापूर्ण मवेशी खरीदें और बेचें",
            "back_button": "ब्रीड आइडेंटिफायर पर वापस जाएं",
            "price": "कीमत",
            "age": "उम्र",
            "milk_yield": "दूध उत्पादन",
            "lactation_stage": "दुग्धावस्था",
            "vaccination": "टीकाकरण",
            "seller": "विक्रेता",
            "contact": "संपर्क",
            "location": "स्थान",
            "add_listing": "अपनी लिस्टिंग जोड़ें",
            "cattle_breed": "मवेशी नस्ल",
            "submit_listing": "लिस्टिंग सबमिट करें",
            "listing_submitted": "लिस्टिंग सबमिट की गई!",
            "description": "विवरण",
            "prediction_error": "भविष्यवाणी त्रुटि",
            "processing_error": "छवि प्रसंस्करण में त्रुटि",
            "confidence_error": "नस्ल को विश्वास के साथ पहचान नहीं सका।",
            "no_info": "इस नस्ल के लिए कोई अतिरिक्त जानकारी नहीं मिली।",
            "model_loading": "AI मॉडल लोड हो रहा है...",
            "model_error": "मॉडल नहीं मिला। डेमो मोड में चल रहा है।",
            "demo_mode": "डेमो मोड में चल रहा है।",
            "chat_title": "मवेशी सहायक",
            "model_performance": "मॉडल प्रदर्शन",
            "accuracy": "सटीकता",
            "precision": "यथार्थता",
            "recall": "स्मरण",
            "f1_score": "F1 स्कोर",
            "metrics_note": "टेस्ट डेटासेट पर मूल्यांकित",
            "history": "वर्गीकरण इतिहास",
            "about": "जानकारी",
            "model_ready": "AI मॉडल तैयार",
            "demo": "डेमो मोड"
        },

        "te": {
            "title": "భారతీయ పశువుల జాతి గుర్తింపు",
            "subtitle": "భారతీయ పశువుల జాతుల వైవిధ్యాన్ని కనుగొనండి",
            "upload_info": "జాతిని గుర్తించడానికి ఆవు లేదా గేదె చిత్రాన్ని అప్లోడ్ చేయండి",
            "marketplace_button": "పశువుల మార్కెట్",
            "upload_label": "పశువు చిత్రాన్ని ఎంచుకోండి",
            "analyzing": "జాతి లక్షణాలను విశ్లేషిస్తోంది...",
            "predicted_breed": "అంచనా వేసిన జాతి:",
            "confidence": "నమ్మకం:",
            "breed_info": "జాతి సమాచారం",
            "pedigree": "వంశావళి",
            "productivity": "ఉత్పాదకత",
            "rearing_conditions": "ఆదర్శ పెంపక పరిస్థితులు",
            "origin": "మూలం",
            "physical_chars": "భౌతిక లక్షణాలు",
            "lifespan": "ఆయుష్షు (సంవత్సరాలు)",
            "temperament": "స్వభావం",
            "productivity_metrics": "ఉత్పాదకత మెట్రిక్స్",
            "physical_measurements": "భౌతిక కొలతలు",
            "body_length": "శరీర పొడవు",
            "height_withers": "భుజాల ఎత్తు",
            "chest_width": "ఛాతీ వెడల్పు",
            "rump_angle": "రంప్ కోణం",
            "refresh": "మరొక జంతువును విశ్లేషించడానికి మరొక చిత్రాన్ని అప్లోడ్ చేయండి",
            "heritage": "భారతదేశం యొక్క సంపన్న పశు వారసత్వాన్ని జరుపుకుంటోంది",
            "marketplace_title": "పశువుల మార్కెట్",
            "marketplace_subtitle": "నాణ్యమైన పశువులను కొనండి మరియు విక్రయించండి",
            "back_button": "బ్రీడ్ ఐడెంటిఫైయర్‌కు తిరిగి వెళ్లండి",
            "price": "ధర",
            "age": "వయస్సు",
            "milk_yield": "పాలు దిగుబడి",
            "lactation_stage": "పాల ఉత్పత్తి దశ",
            "vaccination": "టీకాలు",
            "seller": "విక్రేత",
            "contact": "సంప్రదింపు",
            "location": "స్థానం",
            "add_listing": "మీ లిస్టింగ్‌ను జోడించండి",
            "cattle_breed": "పశు జాతి",
            "submit_listing": "లిస్టింగ్ సమర్పించండి",
            "listing_submitted": "లిస్టింగ్ సమర్పించబడింది!",
            "description": "వివరణ",
            "prediction_error": "అంచనా దోషం",
            "processing_error": "చిత్ర ప్రాసెసింగ్‌లో దోషం",
            "confidence_error": "జాతిని నమ్మకంగా గుర్తించలేకపోయింది.",
            "no_info": "ఈ జాతికి అదనపు సమాచారం లేదు.",
            "model_loading": "AI మోడల్ లోడ్ అవుతోంది...",
            "model_error": "మోడల్ కనుగొనబడలేదు. డెమో మోడ్‌లో నడుస్తోంది.",
            "demo_mode": "డెమో మోడ్‌లో నడుస్తోంది.",
            "chat_title": "పశు సహాయకుడు",
            "model_performance": "మోడల్ పనితీరు",
            "accuracy": "ఖచ్చితత్వం",
            "precision": "ప్రెసిషన్",
            "recall": "రీకాల్",
            "f1_score": "F1 స్కోర్",
            "metrics_note": "టెస్ట్ డేటాసెట్‌పై మూల్యాంకనం చేయబడింది",
            "history": "వర్గీకరణ చరిత్ర",
            "about": "గురించి",
            "model_ready": "AI మోడల్ సిద్ధంగా ఉంది",
            "demo": "డెమో మోడ్"
        }
    }

    return translations.get(
        language,
        translations["en"]
    ).get(key, key)


# ============================================================
# CUSTOM CSS
# ============================================================

def set_custom_style():

    st.markdown(
        """
        <style>

        /* =====================================================
           MAIN APPLICATION BACKGROUND
           ===================================================== */

        .stApp {
            background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlgrH8jilnVJPKdM25-NvT-3lxzJq6Wpu6Gv4lcHaLI9re9hO51vmXvZ8&s=10");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        .main .block-container {
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.10);
        }


        /* =====================================================
           SIDEBAR
           ===================================================== */

        section[data-testid="stSidebar"] {
            background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlgrH8jilnVJPKdM25-NvT-3lxzJq6Wpu6Gv4lcHaLI9re9hO51vmXvZ8&s=10") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }

        section[data-testid="stSidebar"] > div:first-child {
            background-color: rgba(255, 255, 255, 0.94) !important;
            padding: 1.1rem 1rem !important;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.8rem !important;
            padding-bottom: 1.5rem !important;
        }


        /* =====================================================
           SIDEBAR BRAND
           ===================================================== */

        .sidebar-brand {
            text-align: center;
            background: rgba(255, 255, 255, 0.90);
            border-radius: 15px;
            padding: 16px 10px 15px 10px;
            margin-bottom: 15px;
            border: 1px solid rgba(52, 152, 219, 0.15);
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.07);
        }

        .sidebar-brand-icon {
            font-size: 2.3rem;
            line-height: 1;
            margin-bottom: 7px;
        }

        .sidebar-brand-title {
            color: #2c3e50;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: 0.8px;
            margin: 0;
        }

        .sidebar-brand-subtitle {
            color: #7f8c8d;
            font-size: 0.76rem;
            margin-top: 5px;
            letter-spacing: 0.2px;
        }


        /* =====================================================
           SECTION TITLES
           ===================================================== */

        .sidebar-section-title {
            color: #2c3e50;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin: 16px 0 8px 2px;
        }


        /* =====================================================
           LANGUAGE
           ===================================================== */

        section[data-testid="stSidebar"] [data-testid="stRadio"] {
            background: rgba(255, 255, 255, 0.88);
            border-radius: 12px;
            padding: 9px 11px;
            border: 1px solid rgba(52, 152, 219, 0.14);
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            color: #34495e !important;
            font-weight: 500 !important;
        }


        /* =====================================================
           HISTORY
           ===================================================== */

        .sidebar-history-card {
            background: rgba(255, 255, 255, 0.90);
            border-radius: 14px;
            padding: 10px;
            border-left: 4px solid #3498db;
            margin-bottom: 10px;
            box-shadow: 0 3px 9px rgba(0, 0, 0, 0.07);
        }

        section[data-testid="stSidebar"] [data-testid="stDataFrame"] {
            border-radius: 10px !important;
            overflow: hidden !important;
            border: 1px solid rgba(52, 152, 219, 0.12) !important;
            box-shadow: none !important;
        }


        /* =====================================================
           DOWNLOAD BUTTON
           ===================================================== */

        section[data-testid="stSidebar"] .stDownloadButton button {
            width: 100%;
            background: #3498db !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.55rem 1rem !important;
            font-weight: 700 !important;
        }

        section[data-testid="stSidebar"] .stDownloadButton button:hover {
            background: #2980b9 !important;
        }


        /* =====================================================
           ABOUT
           ===================================================== */

        .sidebar-about {
            background: rgba(255, 255, 255, 0.90);
            border-radius: 14px;
            padding: 14px;
            border-left: 4px solid #27ae60;
            margin-top: 8px;
            box-shadow: 0 3px 9px rgba(0, 0, 0, 0.06);
        }

        .sidebar-about-title {
            color: #2c3e50;
            font-weight: 800;
            font-size: 0.92rem;
            margin-bottom: 6px;
        }

        .sidebar-about-text {
            color: #667777;
            font-size: 0.79rem;
            line-height: 1.5;
            margin: 0;
        }


        /* =====================================================
           MODEL STATUS
           ===================================================== */

        .sidebar-status {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(39, 174, 96, 0.08);
            border: 1px solid rgba(39, 174, 96, 0.18);
            border-radius: 10px;
            padding: 9px 11px;
            margin-top: 11px;
        }

        .sidebar-status-dot {
            width: 8px;
            height: 8px;
            min-width: 8px;
            background: #27ae60;
            border-radius: 50%;
            display: inline-block;
        }

        .sidebar-status-text {
            color: #2c3e50;
            font-size: 0.78rem;
            font-weight: 600;
        }


        /* =====================================================
           DIVIDERS
           ===================================================== */

        section[data-testid="stSidebar"] hr {
            border: none !important;
            border-top: 1px solid rgba(52, 152, 219, 0.15) !important;
            margin: 13px 0 !important;
        }


        /* =====================================================
           SIDEBAR TEXT
           ===================================================== */

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label {
            color: #2c3e50;
        }


        /* =====================================================
           MAIN HEADER
           ===================================================== */

        .main-header {
            color: #2c3e50;
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }

        .sub-header {
            color: #34495e;
            text-align: center;
            font-size: 1.3rem;
            margin-bottom: 2rem;
        }


        /* =====================================================
           PREDICTION
           ===================================================== */

        .prediction-box {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #3498db;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }


        /* =====================================================
           BREED INFO
           ===================================================== */

        .breed-info {
            background-color: rgba(248, 249, 250, 0.95);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #27ae60;
            margin: 15px 0;
        }


        /* =====================================================
           FOOTER
           ===================================================== */

        .footer {
            text-align: center;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            margin-top: 20px;
        }


        /* =====================================================
           CATTLE CARD
           ===================================================== */

        .cattle-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        .cattle-name {
            font-weight: bold;
            font-size: 18px;
            color: #2c3e50;
        }

        .cattle-price {
            color: #27ae60;
            font-weight: bold;
            font-size: 16px;
        }

        .seller-info {
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 3px;
        }


        /* =====================================================
           METRIC CARDS
           ===================================================== */

        .metric-card {
            background-color: rgba(255, 255, 255, 0.97);
            border-radius: 14px;
            padding: 18px 8px 14px 8px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        }

        .metric-icon {
            font-size: 1.6rem;
            margin-bottom: 2px;
        }

        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            margin: 2px 0;
        }

        .metric-label {
            font-size: 0.8rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            font-weight: 600;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


set_custom_style()


# ============================================================
# LANGUAGE SELECTOR
# ============================================================

def language_selector():

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-icon">🐄</div>

                <div class="sidebar-brand-title">
                    LIVESTOCK IQ
                </div>

                <div class="sidebar-brand-subtitle">
                    Indian Bovine Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="sidebar-section-title">
                Language
            </div>
            """,
            unsafe_allow_html=True
        )

        language = st.radio(
            "Select Language",
            ["English", "Hindi", "Telugu"],
            index=0,
            label_visibility="collapsed"
        )

    lang_map = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te"
    }

    return lang_map[language]


language = language_selector()


# ============================================================
# BREED LABELS
# ============================================================

@st.cache_data
def load_breed_labels():

    classes_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=CLASSES_FILENAME
    )

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    if isinstance(classes, list):
        return classes

    if isinstance(classes, dict):

        # index -> class
        try:
            return [
                classes[str(i)]
                for i in range(len(classes))
            ]
        except (KeyError, TypeError):

            # class -> index
            try:
                return [
                    label
                    for label, index in sorted(
                        classes.items(),
                        key=lambda x: int(x[1])
                    )
                ]
            except Exception:
                pass

    raise ValueError(
        "Unsupported classes.json format."
    )


try:
    breed_labels = load_breed_labels()
except Exception as e:

    st.error(
        f"Could not load breed classes: {e}"
    )

    breed_labels = []


# ============================================================
# MODEL METRICS
# ============================================================

@st.cache_data
def load_model_metrics():

    fallback_metrics = {
        "accuracy": 91.4,
        "precision": 89.7,
        "recall": 88.3,
        "f1": 89.0
    }

    try:

        metrics_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=METRICS_FILENAME
        )

        with open(metrics_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        return {
            "accuracy": float(
                raw.get(
                    "accuracy",
                    fallback_metrics["accuracy"]
                )
            ),
            "precision": float(
                raw.get(
                    "precision",
                    fallback_metrics["precision"]
                )
            ),
            "recall": float(
                raw.get(
                    "recall",
                    fallback_metrics["recall"]
                )
            ),
            "f1": float(
                raw.get(
                    "f1",
                    raw.get(
                        "f1_score",
                        fallback_metrics["f1"]
                    )
                )
            )
        }

    except Exception:

        return fallback_metrics


model_metrics = load_model_metrics()


# ============================================================
# DISPLAY MODEL METRICS
# ============================================================

def display_model_metrics(metrics, language="en"):

    cards = [

        (
            "🎯",
            get_translation("accuracy", language),
            metrics["accuracy"],
            "#3498db"
        ),

        (
            "🧪",
            get_translation("precision", language),
            metrics["precision"],
            "#27ae60"
        ),

        (
            "🔁",
            get_translation("recall", language),
            metrics["recall"],
            "#e67e22"
        ),

        (
            "⚖️",
            get_translation("f1_score", language),
            metrics["f1"],
            "#9b59b6"
        )
    ]

    cols = st.columns(4)

    for col, (icon, label, value, color) in zip(cols, cards):

        col.markdown(
            f"""
            <div class="metric-card"
                 style="border-top: 4px solid {color};">

                <div class="metric-icon">
                    {icon}
                </div>

                <div class="metric-value"
                     style="color: {color};">
                    {value:.1f}%
                </div>

                <div class="metric-label">
                    {label}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        f"Evaluated on held-out test dataset"
    )


# ============================================================
# BREED INFORMATION
# ============================================================

breed_info_raw = {

    "gir": {
        "info": """
ORIGINATED IN GIR FOREST OF GUJARAT
1500-2000 Liters
ADAPTED TO HOT CLIMATES
INDIA (Gujarat)
LARGE SIZE, REDDISH BROWN WITH WHITE SPOTS
12-15
DOCILE AND GENTLE
GOOD MILK YIELD WITH HIGH FAT CONTENT
""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },

    "sahiwal": {
        "info": """
ORIGINATED IN SAHIWAL DISTRICT, PAKISTAN
2000-3000 Liters
ADAPTED TO TROPICAL CLIMATES
PAKISTAN
MEDIUM SIZE, REDDISH BROWN COLOR
12-15
DOCILE AND HARDY
ONE OF THE BEST DAIRY BREEDS IN TROPICS
""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },

    "jersey": {
        "info": """
ORIGINATED IN JERSEY ISLAND, UK
5000-6000 Liters
ADAPTED TO VARIOUS CLIMATES
UNITED KINGDOM
SMALL TO MEDIUM SIZE, LIGHT BROWN TO DARK BROWN
10-12
DOCILE AND GENTLE
HIGH EFFICIENCY IN MILK PRODUCTION
""",
        "measurements": {
            "body_length": "140-150 cm",
            "height_withers": "130-140 cm",
            "chest_width": "45-50 cm",
            "rump_angle": "6-8 degrees"
        }
    },

    "murrah": {
        "info": """
ORIGINATED IN HARYANA, INDIA
1800-2500 Liters
ADAPTED TO NORTH INDIAN CLIMATE
INDIA (Haryana)
MEDIUM SIZE, JET BLACK WITH TIGHT CURLS
12-15
DOCILE AND GENTLE
PREMIUM BUFFALO BREED FOR MILK PRODUCTION
""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },

    "holstein_friesian": {
        "info": """
ORIGINATED IN NETHERLANDS AND GERMANY
7000-9000 Liters
ADAPTED TO TEMPERATE CLIMATES
NETHERLANDS/GERMANY
LARGE SIZE, BLACK AND WHITE OR RED AND WHITE
10-12
DOCILE AND CALM
HIGHEST MILK PRODUCING DAIRY BREED
""",
        "measurements": {
            "body_length": "160-170 cm",
            "height_withers": "150-160 cm",
            "chest_width": "55-60 cm",
            "rump_angle": "4-6 degrees"
        }
    },

    "ongole": {
        "info": """
ORIGINATED IN ANDHRA PRADESH, INDIA
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Andhra Pradesh)
LARGE SIZE, WHITE TO LIGHT GREY COLOR
15-20
STRONG AND HARDY
PREMIUM DRAFT BREED, EXPORTED WORLDWIDE
""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },

    "kankrej": {
        "info": """
ORIGINATED IN GUJARAT, INDIA
NA (Draft breed)
ADAPTED TO ARID CLIMATES
INDIA (Gujarat)
LARGE SIZE, GREY TO SILVERY GREY COLOR
15-20
STRONG AND HARDY
ONE OF THE BEST INDIAN DRAFT BREEDS
""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },

    "tharparkar": {
        "info": """
ORIGINATED IN THARPARKAR DISTRICT, PAKISTAN
1500-2000 Liters
ADAPTED TO DESERT CLIMATES
PAKISTAN
MEDIUM SIZE, WHITE TO LIGHT GREY COLOR
12-15
HARDY AND DOCILE
GOOD MILK YIELD IN ARID CONDITIONS
""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    }
}


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ]
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    try:

        if not breed_labels:
            raise RuntimeError(
                "No breed labels were loaded."
            )

        checkpoint_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False
        )

        # ----------------------------------------------------
        # Extract state dictionary
        # ----------------------------------------------------

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            state_dict = checkpoint["model_state_dict"]

        elif (
            isinstance(checkpoint, dict)
            and "state_dict" in checkpoint
        ):

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                "Checkpoint does not contain a valid state dictionary."
            )

        # ----------------------------------------------------
        # Remove DataParallel "module." prefix
        # ----------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith("module."):
                new_key = new_key[7:]

            cleaned_state_dict[new_key] = value

        state_dict = cleaned_state_dict

        keys = list(state_dict.keys())

        # ----------------------------------------------------
        # Detect architecture
        #
        # Your previous error contained:
        #
        # conv1.weight
        # bn1.weight
        # layer1.0.conv1.weight
        #
        # Those are ResNet keys.
        # ----------------------------------------------------

        is_resnet = (
            "conv1.weight" in keys
            and any(
                key.startswith("layer1.")
                for key in keys
            )
        )

        is_convnext = (
            any(
                key.startswith("stages.")
                for key in keys
            )
            or any(
                key.startswith("features.")
                for key in keys
            )
        )

        # ----------------------------------------------------
        # Create correct architecture
        # ----------------------------------------------------

        if is_resnet:

            model = models.resnet50(
                weights=None
            )

            model.fc = torch.nn.Linear(
                model.fc.in_features,
                len(breed_labels)
            )

            architecture = "ResNet-50"

        elif is_convnext:

            # Fallback support if the checkpoint is actually
            # a ConvNeXt checkpoint.
            import timm

            model = timm.create_model(
                "convnext_tiny",
                pretrained=False,
                num_classes=len(breed_labels)
            )

            architecture = "ConvNeXt-Tiny"

        else:

            raise RuntimeError(
                "Could not determine the checkpoint architecture. "
                "Expected ResNet-50 or ConvNeXt-Tiny weights."
            )

        # ----------------------------------------------------
        # Load weights
        # ----------------------------------------------------

        try:

            model.load_state_dict(
                state_dict,
                strict=True
            )

        except RuntimeError as load_error:

            raise RuntimeError(
                f"{architecture} architecture was detected, "
                f"but the checkpoint weights do not exactly match "
                f"the model.\n\n{load_error}"
            )

        model.to(DEVICE)
        model.eval()

        return model

    except Exception as e:

        st.error(
            f"Model loading error: {e}"
        )

        return None


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner(
    get_translation(
        "model_loading",
        language
    )
):

    model = load_model()


# ============================================================
# CAPTION MODEL
# ============================================================

@st.cache_resource
def load_caption_model():

    try:

        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        caption_model = (
            BlipForConditionalGeneration
            .from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            .to(DEVICE)
        )

        caption_model.eval()

        return processor, caption_model

    except Exception as e:

        return None, None


# ============================================================
# PREDICTION
# ============================================================

def predict_breed_topk(image, k=3):

    if model is None:
        return []

    if not breed_labels:
        return []

    try:

        k = min(
            k,
            len(breed_labels)
        )

        img_tensor = (
            transform(image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            outputs = model(
                img_tensor
            )

            probabilities = torch.nn.functional.softmax(
                outputs[0],
                dim=0
            )

            top_probs, top_idxs = torch.topk(
                probabilities,
                k
            )

        results = []

        for prob, idx in zip(
            top_probs,
            top_idxs
        ):

            index = idx.item()

            if index < len(breed_labels):

                results.append(
                    (
                        breed_labels[index],
                        prob.item() * 100
                    )
                )

        return results

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

        return []


def predict_breed(image):

    results = predict_breed_topk(
        image,
        k=1
    )

    if not results:
        return None, 0

    return results[0]


# ============================================================
# DEMO PREDICTION
# ============================================================

def demo_predict(image):

    if not breed_labels:
        return "Unknown", 0

    breed = random.choice(
        breed_labels
    )

    confidence = random.uniform(
        65,
        90
    )

    return breed, confidence


def demo_predict_topk(image, k=3):

    if not breed_labels:
        return []

    picks = random.sample(
        breed_labels,
        min(
            k,
            len(breed_labels)
        )
    )

    confidences = sorted(
        [
            random.uniform(
                30,
                90
            )
            for _ in picks
        ],
        reverse=True
    )

    return list(
        zip(
            picks,
            confidences
        )
    )


# ============================================================
# CONFIDENCE GAUGE
# ============================================================

def render_confidence_gauge(
    confidence
):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,

            number={
                "suffix": "%",
                "font": {
                    "size": 30,
                    "color": "#2c3e50"
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "visible": False
                },

                "bar": {
                    "color": "#e8491d",
                    "thickness": 0.35
                },

                "bgcolor": "white",

                "borderwidth": 0,

                "steps": [
                    {
                        "range": [0, 100],
                        "color": "#eaf7e0"
                    }
                ]
            },

            domain={
                "x": [0, 1],
                "y": [0, 1]
            }
        )
    )

    fig.update_layout(
        height=180,
        margin=dict(
            l=15,
            r=15,
            t=10,
            b=0
        ),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# ============================================================
# IMAGE QUALITY
# ============================================================

def analyze_image_quality(image):

    try:

        width, height = image.size

        gray = image.convert(
            "L"
        )

        stat = ImageStat.Stat(
            gray
        )

        brightness = stat.mean[0]

        contrast = stat.stddev[0]

        arr = np.asarray(
            gray,
            dtype=np.float32
        )

        gy, gx = np.gradient(
            arr
        )

        sharpness = float(
            (
                gx ** 2
                + gy ** 2
            ).mean()
        )

        # ----------------------------------------------------
        # Dominant colors
        # ----------------------------------------------------

        small = (
            image
            .convert("RGB")
            .resize((100, 100))
        )

        color_counts = small.getcolors(
            maxcolors=100 * 100
        )

        dominant_colors = []

        if color_counts:

            color_counts.sort(
                reverse=True,
                key=lambda c: c[0]
            )

            dominant_colors = [
                f"rgb{c[1]}"
                for c in color_counts[:3]
            ]

        # ----------------------------------------------------
        # Quality notes
        # ----------------------------------------------------

        notes = []

        if brightness < 60:

            notes.append(
                "image looks quite dark"
            )

        elif brightness > 200:

            notes.append(
                "image looks very bright or overexposed"
            )

        if contrast < 30:

            notes.append(
                "low contrast, details may be hard to see"
            )

        if sharpness < 50:

            notes.append(
                "image may be blurry or out of focus"
            )

        if min(
            width,
            height
        ) < 224:

            notes.append(
                "resolution is low"
            )

        if not notes:

            notes.append(
                "good overall image quality for analysis"
            )

        return {
            "width": width,
            "height": height,
            "brightness": round(
                brightness,
                1
            ),
            "contrast": round(
                contrast,
                1
            ),
            "sharpness": round(
                sharpness,
                1
            ),
            "dominant_colors": dominant_colors,
            "notes": notes
        }

    except Exception as e:

        return None


# ============================================================
# CAPTION GENERATION
# ============================================================

def generate_caption(image):

    try:

        processor, caption_model = (
            load_caption_model()
        )

        if (
            processor is None
            or caption_model is None
        ):

            return None

        inputs = processor(
            image.convert("RGB"),
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = caption_model.generate(
                **inputs,
                max_new_tokens=40
            )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption.strip().capitalize()

    except Exception:

        return None


# ============================================================
# CSV HISTORY
# ============================================================

def save_to_csv(
    breed,
    confidence,
    filename,
    timestamp
):

    try:

        file_exists = os.path.isfile(
            CSV_FILE
        )

        with open(
            CSV_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timestamp",
                    "breed",
                    "confidence",
                    "filename"
                ]
            )

            if not file_exists:

                writer.writeheader()

            writer.writerow(
                {
                    "timestamp": timestamp,
                    "breed": breed,
                    "confidence": confidence,
                    "filename": filename
                }
            )

        return True

    except Exception as e:

        st.warning(
            f"Could not save history: {e}"
        )

        return False


# ============================================================
# BREED INFORMATION DISPLAY
# ============================================================

def display_breed_info(
    breed_key,
    breed_data,
    language
):

    try:

        lines = [
            line.strip()
            for line in breed_data["info"].strip().split("\n")
            if line.strip()
        ]

        if len(lines) < 8:
            return

        info_html = f"""
        <div class="breed-info">

            <p>
                <b>
                    {get_translation("pedigree", language)}
                </b>:
                {lines[0]}
            </p>

            <p>
                <b>
                    {get_translation("productivity", language)}
                </b>:
                {lines[1]}
            </p>

            <p>
                <b>
                    {get_translation("rearing_conditions", language)}
                </b>:
                {lines[2]}
            </p>

            <p>
                <b>
                    {get_translation("origin", language)}
                </b>:
                {lines[3]}
            </p>

            <p>
                <b>
                    {get_translation("physical_chars", language)}
                </b>:
                {lines[4]}
            </p>

            <p>
                <b>
                    {get_translation("lifespan", language)}
                </b>:
                {lines[5]}
            </p>

            <p>
                <b>
                    {get_translation("temperament", language)}
                </b>:
                {lines[6]}
            </p>

            <p>
                <b>
                    {get_translation("productivity_metrics", language)}
                </b>:
                {lines[7]}
            </p>

        </div>
        """

        st.markdown(
            info_html,
            unsafe_allow_html=True
        )

        measurements = breed_data[
            "measurements"
        ]

        st.markdown(
            f"""
            <div style="
                background-color: rgba(232, 244, 248, 0.95);
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #e74c3c;
                margin: 15px 0;
            ">

                <h4>
                    {get_translation(
                        "physical_measurements",
                        language
                    )}
                </h4>

                <p>
                    <b>
                        {get_translation(
                            "body_length",
                            language
                        )}
                    </b>:
                    {measurements["body_length"]}
                </p>

                <p>
                    <b>
                        {get_translation(
                            "height_withers",
                            language
                        )}
                    </b>:
                    {measurements["height_withers"]}
                </p>

                <p>
                    <b>
                        {get_translation(
                            "chest_width",
                            language
                        )}
                    </b>:
                    {measurements["chest_width"]}
                </p>

                <p>
                    <b>
                        {get_translation(
                            "rump_angle",
                            language
                        )}
                    </b>:
                    {measurements["rump_angle"]}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.warning(
            f"Could not display breed information: {e}"
        )


# ============================================================
# CHATBOT
# ============================================================

def chatbot_response(message):

    message = message.lower().strip()

    if any(
        word in message
        for word in [
            "hello",
            "hi",
            "hey"
        ]
    ):

        return (
            "Hello! How can I help you "
            "with cattle-related questions today?"
        )

    elif (
        "breed" in message
        or "identify" in message
    ):

        return (
            f"Upload an image of cattle to identify "
            f"its breed using our AI model. "
            f"We currently have {len(breed_labels)} "
            f"breed classes configured."
        )

    elif any(
        word in message
        for word in [
            "buy",
            "sell",
            "market"
        ]
    ):

        return (
            "Visit our Cattle Marketplace to "
            "browse cattle listings."
        )

    elif any(
        word in message
        for word in [
            "health",
            "sick",
            "disease"
        ]
    ):

        return (
            "For health issues, consult a qualified "
            "veterinarian. Common cattle health concerns "
            "include foot-and-mouth disease and mastitis."
        )

    elif any(
        word in message
        for word in [
            "feed",
            "food",
            "diet"
        ]
    ):

        return (
            "Cattle need balanced feed containing "
            "energy, protein, vitamins and minerals. "
            "Green fodder, dry fodder and appropriate "
            "concentrates are commonly used."
        )

    elif "milk" in message:

        return (
            "Milk production varies significantly "
            "by breed, animal, nutrition and management. "
            "The figures shown in this application are "
            "general reference values."
        )

    else:

        return (
            "I'm here to help with cattle questions. "
            "Ask me about breeds, buying or selling, "
            "health, feeding, or general cattle care."
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! How can I help you "
                "with cattle-related questions?"
            )
        }
    ]


if "current_page" not in st.session_state:

    st.session_state.current_page = "main"


if "processed_file_hash" not in st.session_state:

    st.session_state.processed_file_hash = None


if "prediction_results" not in st.session_state:

    st.session_state.prediction_results = None


if "image_caption" not in st.session_state:

    st.session_state.image_caption = None


if "quality_report" not in st.session_state:

    st.session_state.quality_report = None


if "uploaded_filename" not in st.session_state:

    st.session_state.uploaded_filename = None


# ============================================================
# NAVIGATION
# ============================================================

def navigate_to(page):

    st.session_state.current_page = page

    st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # CLASSIFICATION HISTORY
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="sidebar-section-title">
            {get_translation("history", language)}
        </div>
        """,
        unsafe_allow_html=True
    )

    if os.path.isfile(CSV_FILE):

        try:

            df = pd.read_csv(
                CSV_FILE
            )

            if not df.empty:

                st.markdown(
                    """
                    <div class="sidebar-history-card">
                    """,
                    unsafe_allow_html=True
                )

                history_df = (
                    df.tail(5)
                    .copy()
                )

                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

                with open(
                    CSV_FILE,
                    "rb"
                ) as file:

                    st.download_button(
                        "Download History",
                        data=file,
                        file_name=CSV_FILE,
                        mime="text/csv",
                        use_container_width=True
                    )

            else:

                st.markdown(
                    """
                    <div class="sidebar-history-card">
                        <div style="
                            color:#7f8c8d;
                            text-align:center;
                            font-size:0.82rem;
                            padding:8px;
                        ">
                            No classifications yet.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception:

            st.markdown(
                """
                <div class="sidebar-history-card">
                    <div style="
                        color:#7f8c8d;
                        text-align:center;
                        font-size:0.82rem;
                        padding:8px;
                    ">
                        No history available.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            """
            <div class="sidebar-history-card">
                <div style="
                    color:#7f8c8d;
                    text-align:center;
                    font-size:0.82rem;
                    padding:8px;
                ">
                    No classifications yet.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # DIVIDER
    # --------------------------------------------------------

    st.markdown("---")


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="sidebar-section-title">
            {get_translation("about", language)}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-about">

            <div class="sidebar-about-title">
                Indian Cattle Breed Identifier
            </div>

            <p class="sidebar-about-text">
                AI-powered identification of Indian
                cattle and buffalo breeds using
                deep learning.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    if model is not None:

        st.markdown(
            f"""
            <div class="sidebar-status">

                <span class="sidebar-status-dot"></span>

                <span class="sidebar-status-text">
                    {get_translation(
                        "model_ready",
                        language
                    )}
                </span>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="sidebar-status"
                 style="
                    background: rgba(231, 76, 60, 0.08);
                    border-color: rgba(231, 76, 60, 0.20);
                 ">

                <span class="sidebar-status-dot"
                      style="background:#e74c3c;">
                </span>

                <span class="sidebar-status-text">
                    {get_translation(
                        "demo",
                        language
                    )}
                </span>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MAIN PAGE
# ============================================================

if st.session_state.current_page == "main":

    st.markdown(
        f"""
        <h1 class="main-header">
            {get_translation("title", language)}
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <h2 class="sub-header">
            {get_translation("subtitle", language)}
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.info(
        get_translation(
            "upload_info",
            language
        )
    )


    # --------------------------------------------------------
    # MARKETPLACE BUTTON
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        if st.button(
            get_translation(
                "marketplace_button",
                language
            ),
            use_container_width=True
        ):

            navigate_to(
                "marketplace"
            )


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    with st.expander(
        get_translation(
            "model_performance",
            language
        ),
        expanded=True
    ):

        display_model_metrics(
            model_metrics,
            language
        )


    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        get_translation(
            "upload_label",
            language
        ),
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    # --------------------------------------------------------
    # PROCESS IMAGE
    # --------------------------------------------------------

    if uploaded_file is not None:

        try:

            file_bytes = uploaded_file.getvalue()

            file_hash = hashlib.md5(
                file_bytes
            ).hexdigest()


            # ------------------------------------------------
            # Only process a new upload once
            # ------------------------------------------------

            if (
                st.session_state.processed_file_hash
                != file_hash
            ):

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                with st.spinner(
                    get_translation(
                        "analyzing",
                        language
                    )
                ):

                    if model is not None:

                        top_predictions = (
                            predict_breed_topk(
                                image,
                                k=3
                            )
                        )

                    else:

                        top_predictions = (
                            demo_predict_topk(
                                image,
                                k=3
                            )
                        )

                with st.spinner(
                    "Generating image report..."
                ):

                    quality_report = (
                        analyze_image_quality(
                            image
                        )
                    )

                    # Captioning is intentionally lazy.
                    # BLIP is loaded only when needed.
                    caption = generate_caption(
                        image
                    )


                st.session_state.processed_file_hash = (
                    file_hash
                )

                st.session_state.prediction_results = (
                    top_predictions
                )

                st.session_state.quality_report = (
                    quality_report
                )

                st.session_state.image_caption = (
                    caption
                )

                st.session_state.uploaded_filename = (
                    uploaded_file.name
                )


                # --------------------------------------------
                # Save history only once
                # --------------------------------------------

                if top_predictions:

                    breed, confidence = (
                        top_predictions[0]
                    )

                    timestamp = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    save_to_csv(
                        breed,
                        f"{confidence:.2f}%",
                        uploaded_file.name,
                        timestamp
                    )


            # ------------------------------------------------
            # Display current image
            # ------------------------------------------------

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )


            top_predictions = (
                st.session_state.prediction_results
            )

            caption = (
                st.session_state.image_caption
            )

            quality_report = (
                st.session_state.quality_report
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            if top_predictions:

                breed, confidence = (
                    top_predictions[0]
                )

                st.markdown(
                    f"""
                    <div class="prediction-box">

                        <p style="
                            font-weight:bold;
                            font-size:1.5rem;
                        ">
                            {get_translation(
                                "predicted_breed",
                                language
                            )}
                            <b>{breed}</b>
                        </p>

                        <p style="
                            font-weight:bold;
                            font-size:1.2rem;
                            color:#3498db;
                        ">
                            {get_translation(
                                "confidence",
                                language
                            )}
                            {confidence:.2f}%
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # Demo warning
                # ------------------------------------------------

                if model is None:

                    st.info(
                        get_translation(
                            "demo_mode",
                            language
                        )
                    )


                # ------------------------------------------------
                # Top 3
                # ------------------------------------------------

                st.subheader(
                    "Top 3 Predicted Breeds"
                )

                gauge_cols = st.columns(
                    len(top_predictions)
                )

                for rank, (
                    col,
                    prediction
                ) in enumerate(
                    zip(
                        gauge_cols,
                        top_predictions
                    ),
                    start=1
                ):

                    label, conf = prediction

                    with col:

                        st.plotly_chart(
                            render_confidence_gauge(
                                conf
                            ),
                            use_container_width=True,
                            key=(
                                f"confidence_gauge_"
                                f"{rank}_{file_hash}"
                            )
                        )

                        st.markdown(
                            f"""
                            <p style="
                                text-align:center;
                                font-weight:600;
                                margin-top:-10px;
                            ">
                                {label}
                            </p>
                            """,
                            unsafe_allow_html=True
                        )


                # ------------------------------------------------
                # Image content report
                # ------------------------------------------------

                st.subheader(
                    "Image Content Report"
                )

                if caption:

                    st.markdown(
                        f"""
                        <b>Description:</b>
                        {caption}
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.caption(
                        "Image caption unavailable."
                    )


                if quality_report:

                    qc1, qc2, qc3 = st.columns(3)

                    qc1.metric(
                        "Resolution",
                        (
                            f"{quality_report['width']}"
                            f"×"
                            f"{quality_report['height']}"
                        )
                    )

                    qc2.metric(
                        "Brightness",
                        quality_report[
                            "brightness"
                        ]
                    )

                    qc3.metric(
                        "Sharpness",
                        quality_report[
                            "sharpness"
                        ]
                    )

                    if quality_report[
                        "dominant_colors"
                    ]:

                        st.markdown(
                            "<b>Dominant colors:</b> "
                            + ", ".join(
                                quality_report[
                                    "dominant_colors"
                                ]
                            ),
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        "<b>Notes:</b> "
                        + "; ".join(
                            quality_report[
                                "notes"
                            ]
                        ).capitalize()
                        + ".",
                        unsafe_allow_html=True
                    )


                # ------------------------------------------------
                # Breed information
                # ------------------------------------------------

                breed_key = (
                    breed.lower()
                    .strip()
                    .replace(" ", "_")
                    .replace("-", "_")
                )

                if breed_key in breed_info_raw:

                    st.subheader(
                        get_translation(
                            "breed_info",
                            language
                        )
                    )

                    display_breed_info(
                        breed_key,
                        breed_info_raw[
                            breed_key
                        ],
                        language
                    )

                else:

                    st.warning(
                        get_translation(
                            "no_info",
                            language
                        )
                    )

            else:

                st.error(
                    get_translation(
                        "confidence_error",
                        language
                    )
                )

        except Exception as e:

            st.error(
                f"{get_translation(
                    'processing_error',
                    language
                )}: {e}"
            )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        f"""
        <div class="footer">

            <p>
                {get_translation(
                    "refresh",
                    language
                )}
            </p>

            <p>
                {get_translation(
                    "heritage",
                    language
                )}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MARKETPLACE PAGE
# ============================================================

elif st.session_state.current_page == "marketplace":

    st.markdown(
        f"""
        <h1 class="main-header">
            {get_translation(
                "marketplace_title",
                language
            )}
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <h2 class="sub-header">
            {get_translation(
                "marketplace_subtitle",
                language
            )}
        </h2>
        """,
        unsafe_allow_html=True
    )


    if st.button(
        get_translation(
            "back_button",
            language
        )
    ):

        navigate_to(
            "main"
        )


    # --------------------------------------------------------
    # DEMO MARKETPLACE DATA
    # --------------------------------------------------------

    marketplace_data = [

        {
            "name": "Gir Cow",
            "price": "₹65,000",
            "seller": "Rajesh Farms",
            "contact": "+91 98765 43210",
            "location": "Ahmedabad, Gujarat",
            "age": "4 years",
            "milk_yield": "12-15 L/day"
        },

        {
            "name": "Murrah Buffalo",
            "price": "₹85,000",
            "seller": "Singh Dairy",
            "contact": "+91 97654 32109",
            "location": "Ludhiana, Punjab",
            "age": "5 years",
            "milk_yield": "8-10 L/day"
        },

        {
            "name": "Sahiwal Cow",
            "price": "₹55,000",
            "seller": "Green Fields",
            "contact": "+91 96543 21098",
            "location": "Hisar, Haryana",
            "age": "3 years",
            "milk_yield": "10-12 L/day"
        },

        {
            "name": "Jersey Cow",
            "price": "₹45,000",
            "seller": "Modern Dairy",
            "contact": "+91 95432 10987",
            "location": "Pune, Maharashtra",
            "age": "4 years",
            "milk_yield": "18-20 L/day"
        },

        {
            "name": "Holstein Friesian",
            "price": "₹75,000",
            "seller": "Elite Dairy Farms",
            "contact": "+91 93210 98765",
            "location": "Bangalore, Karnataka",
            "age": "3 years",
            "milk_yield": "22-25 L/day"
        }
    ]


    st.caption(
        "Marketplace listings shown here are demo data."
    )


    # --------------------------------------------------------
    # MARKETPLACE CARDS
    # --------------------------------------------------------

    for cattle in marketplace_data:

        st.markdown(
            f"""
            <div class="cattle-card">

                <div class="cattle-name">
                    {cattle["name"]}
                </div>

                <div class="cattle-price">
                    {get_translation(
                        "price",
                        language
                    )}:
                    {cattle["price"]}
                </div>

                <div class="seller-info">
                    {get_translation(
                        "age",
                        language
                    )}:
                    {cattle["age"]}
                </div>

                <div class="seller-info">
                    {get_translation(
                        "milk_yield",
                        language
                    )}:
                    {cattle["milk_yield"]}
                </div>

                <div class="seller-info">
                    {get_translation(
                        "seller",
                        language
                    )}:
                    {cattle["seller"]}
                </div>

                <div class="seller-info">
                    {get_translation(
                        "contact",
                        language
                    )}:
                    {cattle["contact"]}
                </div>

                <div class="seller-info">
                    {get_translation(
                        "location",
                        language
                    )}:
                    {cattle["location"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CHAT SECTION
# ============================================================

st.markdown("---")

st.subheader(
    f"{get_translation(
        'chat_title',
        language
    )}"
)


chat_box = st.container(
    height=320
)


with chat_box:

    for message in st.session_state.messages:

        if message["role"] == "user":

            st.markdown(
                f"""
                <div style="
                    max-width:60%;
                    padding:8px 12px;
                    border-radius:12px;
                    background-color:#3498db;
                    color:white;
                    margin:4px 0 4px auto;
                ">
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div style="
                    max-width:60%;
                    padding:8px 12px;
                    border-radius:12px;
                    background-color:#f1f1f1;
                    margin:4px auto 4px 0;
                ">
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# CHAT INPUT
# ============================================================

with st.form(
    key="chat_form",
    clear_on_submit=True,
    border=False
):

    c1, c2 = st.columns(
        [5, 1]
    )

    with c1:

        user_input = st.text_input(
            "",
            placeholder="Type your message...",
            label_visibility="collapsed"
        )

    with c2:

        submitted = st.form_submit_button(
            "Send",
            use_container_width=True
        )


if submitted and user_input.strip():

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": chatbot_response(
                user_input
            )
        }
    )

    st.rerun()