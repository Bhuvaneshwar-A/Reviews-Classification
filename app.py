import streamlit as st
from PIL import Image
import base64
import json
import torch
from inference.pipeline import InferencePipeline
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import time

st.set_page_config(page_title="Product Review Classification", layout="wide", initial_sidebar_state="collapsed")

css = '''
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.header-container {
    padding: 2rem 0;
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.header-container h1 {
    font-weight: 600;
    margin-bottom: 0.5rem;
    font-size: 2.5rem;
}

.header-container p {
    opacity: 0.9;
    font-size: 1.1rem;
}

.info-container {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.info-container h2 {
    color: #1a1a1a;
    margin-bottom: 1rem;
    font-weight: 600;
}

.info-container ul {
    color: #4a4a4a;
    padding-left: 1.5rem;
}

.info-container li {
    margin-bottom: 0.5rem;
}

.stTextArea textarea {
    border-radius: 10px;
    border: 2px solid #e0e0e0;
    padding: 1rem;
    font-size: 1rem;
    transition: all 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
}

.results-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1a1a1a;
    text-align: center;
    margin: 2rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #667eea;
}

.chart-container {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.predictions-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
}

.prediction-item {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    position: relative;
    overflow: hidden;
}

.prediction-item .category {
    font-weight: 500;
    margin-bottom: 0.5rem;
    color: #1a1a1a;
}

.confidence-bar {
    height: 4px;
    background: #4CAF50;
    border-radius: 2px;
    margin-bottom: 0.5rem;
    transition: width 1s ease-in-out;
}

.confidence-value {
    font-size: 0.9rem;
    color: #666;
    text-align: right;
}

.footer {
    margin-top: 3rem;
    text-align: center;
    padding: 1rem;
    color: #666;
    font-size: 0.9rem;
}

.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 50px;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}

.stButton button:active {
    transform: translateY(0);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

[data-testid="stMarkdownContainer"] {
    width: 100%;
}
'''
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

# Fallback Lottie JSON data
LOTTIE_ANALYZE = {
    "v": "5.7.4",
    "fr": 30,
    "ip": 0,
    "op": 60,
    "w": 512,
    "h": 512,
    "nm": "Analysis",
    "ddd": 0,
    "assets": [],
    "layers": [{
        "ddd": 0,
        "ind": 1,
        "ty": 4,
        "nm": "Circle",
        "sr": 1,
        "ks": {
            "o": {"a": 0, "k": 100},
            "r": {"a": 1, "k": [{"t": 0, "s": [0]}, {"t": 60, "s": [360]}]},
            "p": {"a": 0, "k": [256, 256]},
            "a": {"a": 0, "k": [0, 0]},
            "s": {"a": 0, "k": [100, 100]}
        },
        "shapes": [{
            "ty": "el",
            "p": {"a": 0, "k": [0, 0]},
            "s": {"a": 0, "k": [200, 200]},
            "c": {"a": 0, "k": [0.4, 0.47, 0.92]}
        }]
    }]
}

LOTTIE_SUCCESS = {
    "v": "5.7.4",
    "fr": 30,
    "ip": 0,
    "op": 60,
    "w": 512,
    "h": 512,
    "nm": "Success",
    "ddd": 0,
    "assets": [],
    "layers": [{
        "ddd": 0,
        "ind": 1,
        "ty": 4,
        "nm": "Checkmark",
        "sr": 1,
        "ks": {
            "o": {"a": 0, "k": 100},
            "r": {"a": 0, "k": 0},
            "p": {"a": 0, "k": [256, 256]},
            "a": {"a": 0, "k": [0, 0]},
            "s": {"a": 1, "k": [
                {"t": 0, "s": [0, 0]},
                {"t": 30, "s": [100, 100]}
            ]}
        },
        "shapes": [{
            "ty": "el",
            "p": {"a": 0, "k": [0, 0]},
            "s": {"a": 0, "k": [200, 200]},
            "c": {"a": 0, "k": [0.3, 0.69, 0.31]}
        }]
    }]
}

# Try loading from URL, fallback to local JSON
# lottie_analyze = load_lottie_url("https://lottie.host/18f1b44d-31ed-43fc-af56-6015f24812da/ReSZJlN11o.json")
# if not lottie_analyze:
#     lottie_analyze = LOTTIE_ANALYZE

lottie_success = load_lottie_url("https://lottie.host/18f1b44d-31ed-43fc-af56-6015f24812da/ReSZJlN11o.json")
if not lottie_success:
    lottie_success = LOTTIE_SUCCESS

def create_radar_chart(predictions):
    categories = [p['category'] for p in predictions]
    confidence = [p['confidence'] * 100 for p in predictions]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=confidence,
        theta=categories,
        fill='toself',
        line_color='#4CAF50',
        fillcolor='rgba(76, 175, 80, 0.5)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    return fig

@st.cache_resource
def load_model():
    return InferencePipeline('best_model.pt')

pipeline = load_model()

st.markdown("""
    <div class="header-container">
        <h1>Product Review Analyzer</h1>
        <p>Advanced ML-powered review analysis using state-of-the-art BERT architecture</p>
    </div>
""", unsafe_allow_html=True)

# col1, col2 = st.columns([2, 1])

# with col1:
#     st_lottie(lottie_analyze, height=300, key="analyze", speed=1.5)

# with col2:
#     st.markdown("""
#         <div class="info-container">
#             <h2>How it works</h2>
#             <ul>
#                 <li>Enter your product review text</li>
#                 <li>AI analyzes sentiment and features</li>
#                 <li>Get instant visual insights</li>
#             </ul>
#         </div>
#     """, unsafe_allow_html=True)

text_input = st.text_area("Enter Product Review", height=150, 
                         placeholder="Type or paste your product review here...")

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    analyze_button = st.button("Analyze Review", use_container_width=True)

if analyze_button and text_input:
    with st.spinner('Analyzing...'):
        predictions = pipeline.predict(text_input)
        time.sleep(0.5)  # Short delay for visual feedback
        st_lottie(lottie_success, height=200, key="success", speed=1.5)
    
    st.markdown("<div class='results-header'>Analysis Results</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.plotly_chart(create_radar_chart(predictions), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='predictions-container'>", unsafe_allow_html=True)
        for pred in sorted(predictions, key=lambda x: x['confidence'], reverse=True):
            confidence = pred['confidence'] * 100
            color = '#4CAF50' if confidence > 75 else '#FFC107' if confidence > 50 else '#F44336'
            
            st.markdown(f"""
                <div class='prediction-item' style='border-left: 4px solid {color};'>
                    <div class='category'>{pred['category']}</div>
                    <div class='confidence-bar' style='width: {confidence}%; background-color: {color};'></div>
                    <div class='confidence-value'>{confidence:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <div class='footer'>
        <p>Powered by BERT & Advanced Deep Learning</p>
    </div>
""", unsafe_allow_html=True)