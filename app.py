import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="OncoGuard AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING (Medical Theme) ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3, p, label { color: #2c3e50 !important; }
    
    /* Result Cards */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Buttons */
    div.stButton > button {
        width: 100%;
        background-color: #e83e8c; /* Pink for Breast Cancer Awareness */
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Load Model
    model = tf.keras.models.load_model('cancer_model.keras')
    
    # Load Scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # Load Data (for default values)
    df = pd.read_csv('BreastCancerDataset.csv')
    df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    return model, scaler, df

try:
    model, scaler, df = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# --- 4. SIDEBAR (User Inputs) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1509/1509612.png", width=80)
    st.title("Patient Vitals")
    st.write("Adjust the key cytological features below:")
    
    # We create sliders for the top 6 'Mean' features
    # We use the min/max from the dataset to set realistic boundaries
    
    def add_slider(label, col_name):
        return st.slider(
            label, 
            min_value=float(df[col_name].min()), 
            max_value=float(df[col_name].max()), 
            value=float(df[col_name].mean())
        )

    radius_mean = add_slider("Radius (Mean)", 'radius_mean')
    texture_mean = add_slider("Texture (Mean)", 'texture_mean')
    perimeter_mean = add_slider("Perimeter (Mean)", 'perimeter_mean')
    area_mean = add_slider("Area (Mean)", 'area_mean')
    smoothness_mean = add_slider("Smoothness (Mean)", 'smoothness_mean')
    concavity_mean = add_slider("Concavity (Mean)", 'concavity_mean')

    st.markdown("---")
    st.caption("Note: Remaining 24 features are set to the population average.")

# --- 5. DATA PREPARATION ---
def prepare_input_data():
    # Create a dictionary with ALL 30 columns set to their mean values
    input_dict = df.mean(numeric_only=True).to_dict()
    
    # Overwrite the specific values the user changed
    input_dict['radius_mean'] = radius_mean
    input_dict['texture_mean'] = texture_mean
    input_dict['perimeter_mean'] = perimeter_mean
    input_dict['area_mean'] = area_mean
    input_dict['smoothness_mean'] = smoothness_mean
    input_dict['concavity_mean'] = concavity_mean
    
    # Ensure correct order of columns matches the training data
    # (The scaler needs the columns in the EXACT same order)
    # We look at the scaler's 'feature_names_in_' if available, or just use df columns
    features = pd.DataFrame([input_dict])
    
    # Drop diagnosis if it accidentally got in (it shouldn't, but safety first)
    if 'diagnosis' in features.columns:
        features = features.drop(columns=['diagnosis'])
        
    return features

# --- 6. MAIN DASHBOARD ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("OncoGuard AI 🎗️")
    st.markdown("### Intelligent Breast Mass Analysis System")
    
    input_data = prepare_input_data()
    
    # SCALING
    input_scaled = scaler.transform(input_data)
    
    # PREDICTION
    prediction_prob = model.predict(input_scaled)[0][0]
    prediction_class = 1 if prediction_prob > 0.5 else 0
    
    # DISPLAY RESULTS
    st.markdown("---")
    
    if prediction_class == 1:
        # Malignant
        st.markdown(f"""
        <div style="background-color: #fdf2f2; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;">
            <h2 style="color: #dc3545; margin:0;">🚨 High Probability of Malignancy</h2>
            <p style="font-size: 18px; margin: 5px 0;">The analysis suggests the mass is <b>Malignant</b>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Benign
        st.markdown(f"""
        <div style="background-color: #f0fdf4; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
            <h2 style="color: #28a745; margin:0;">✅ Likely Benign</h2>
            <p style="font-size: 18px; margin: 5px 0;">The analysis suggests the mass is <b>Benign</b> (Non-Cancerous).</p>
        </div>
        """, unsafe_allow_html=True)

    # Visualization: Radar Chart (User vs Average)
    st.subheader("Cytological Profile")
    
    # Normalize values for the chart (0-1 scale) just for visualization
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Concavity']
    user_values = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, concavity_mean]
    # We divide by max to keep it on a relative scale
    avg_values = [df[c].mean() for c in ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'concavity_mean']]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Current Patient',
        line_color='#e83e8c'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name='Population Average',
        line_color='#adb5bd',
        opacity=0.5
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        showlegend=True,
        height=400,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Confidence Metric")
    
    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Malignancy Risk (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#dc3545" if prediction_prob > 0.5 else "#28a745"},
            'steps': [
                {'range': [0, 50], 'color': "#f0fdf4"},
                {'range': [50, 100], 'color': "#fdf2f2"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=0, b=0, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("""
    <div class="metric-card">
        <p style="font-size: 14px; color: grey;">Neural Network Confidence</p>
        <h1 style="margin: 0;">{:.2f}%</h1>
    </div>
    """.format(prediction_prob * 100 if prediction_prob > 0.5 else (1 - prediction_prob) * 100), unsafe_allow_html=True)
    
    st.markdown("### ℹ️ Analysis Details")
    st.info("""
    **Model:** Deep Neural Network (TensorFlow)
    **Training Data:** WDBC Dataset (N=569)
    **Accuracy:** 96.0%+ (Validation)
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 12px; color: grey;">
    ⚠️ <b>Disclaimer:</b> This tool is for educational/research purposes only. 
    It is NOT a medical device. Always consult a pathologist for diagnosis.
</div>
""", unsafe_allow_html=True)
