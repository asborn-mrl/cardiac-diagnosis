"""
🫀 Intelligent Cardiac Diagnosis System
A Hybrid Approach Using Fuzzy Logic & LightGBM

Streamlit Web Application for Deployment
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from fuzzy_transformer import FuzzyFeatureTransformer

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="Cardiac Diagnosis System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #fadbd8;
        border: 2px solid #e74c3c;
        color: #c0392b;
    }
    .low-risk {
        background-color: #d5f5e3;
        border: 2px solid #27ae60;
        color: #1e8449;
    }
    .medium-risk {
        background-color: #fdebd0;
        border: 2px solid #f39c12;
        color: #d68910;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
    }
    .info-box {
        background-color: #ebf5fb;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)


# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    """Load the trained model and fuzzy transformer."""
    try:
        with open('model/hybrid_cardiac_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['fuzzy_transformer'], model_data['feature_names']
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'model/hybrid_cardiac_model.pkl' exists.")
        return None, None, None


# ============ PREDICTION FUNCTION ============
def predict_heart_disease(patient_data, model, fuzzy_transformer):
    """Make prediction for a patient."""
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Create DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Transform to fuzzy features
    fuzzy_features = fuzzy_transformer.transform(patient_df)
    
    # Combine features
    X = pd.concat([patient_df[categorical_features], fuzzy_features], axis=1)
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return prediction, probability


# ============ VISUALIZATION FUNCTIONS ============
def create_gauge_chart(probability):
    """Create a gauge chart for risk visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Heart Disease Risk (%)", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "#e74c3c"}, 'decreasing': {'color': "#27ae60"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#3498db"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d5f5e3'},
                {'range': [30, 60], 'color': '#fdebd0'},
                {'range': [60, 100], 'color': '#fadbd8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_fuzzy_membership_chart(value, feature_name, ranges, memberships, labels):
    """Create a chart showing fuzzy membership for a feature."""
    fig = go.Figure()
    
    colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
    
    for i, (membership, label) in enumerate(zip(memberships, labels)):
        fig.add_trace(go.Scatter(
            x=ranges,
            y=membership,
            mode='lines',
            name=label,
            line=dict(width=2, color=colors[i % len(colors)]),
            fill='tozeroy',
            fillcolor=f'rgba{tuple(list(int(colors[i % len(colors)][j:j+2], 16) for j in (1, 3, 5)) + [0.1])}'
        ))
    
    # Add vertical line for current value
    fig.add_vline(x=value, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"Patient: {value}", annotation_position="top")
    
    fig.update_layout(
        title=f"{feature_name} Fuzzy Membership",
        xaxis_title=feature_name,
        yaxis_title="Membership Degree",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_feature_comparison_chart(patient_data):
    """Create a radar chart comparing patient features to normal ranges."""
    categories = ['Age', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'ST Depression']
    
    # Normalize values to 0-100 scale for visualization
    patient_values = [
        (patient_data['age'] - 20) / (80 - 20) * 100,
        (patient_data['trestbps'] - 80) / (200 - 80) * 100,
        (patient_data['chol'] - 100) / (400 - 100) * 100,
        (patient_data['thalach'] - 60) / (200 - 60) * 100,
        patient_data['oldpeak'] / 6 * 100
    ]
    
    # Normal ranges (middle values)
    normal_values = [50, 40, 40, 60, 15]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=patient_values + [patient_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Patient',
        line_color='#e74c3c',
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=normal_values + [normal_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Normal Range',
        line_color='#27ae60',
        fillcolor='rgba(39, 174, 96, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


# ============ MAIN APPLICATION ============
def main():
    # Header
    st.markdown('<p class="main-header">🫀 Intelligent Cardiac Diagnosis System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Hybrid Approach Using Fuzzy Logic & LightGBM</p>', unsafe_allow_html=True)
    
    # Load model
    model, fuzzy_transformer, feature_names = load_model()
    
    if model is None:
        st.warning("⚠️ Please train and save the model first using the Jupyter notebook.")
        st.info("📝 After training, save the model to `model/hybrid_cardiac_model.pkl`")
        
        # Show demo mode
        st.markdown("---")
        st.markdown("### 🎮 Demo Mode")
        st.info("Running in demo mode with simulated predictions.")
        demo_mode = True
    else:
        demo_mode = False
    
    # Sidebar - Patient Information Input
    st.sidebar.markdown("## 📋 Patient Information")
    st.sidebar.markdown("---")
    
    # Personal Information
    st.sidebar.markdown("### 👤 Personal Details")
    age = st.sidebar.slider("Age (years)", 20, 90, 55, help="Patient's age in years")
    sex = st.sidebar.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
    
    # Clinical Measurements
    st.sidebar.markdown("### 🏥 Clinical Measurements")
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130, 
                                  help="Resting blood pressure on admission")
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 240,
                             help="Serum cholesterol level")
    thalach = st.sidebar.slider("Max Heart Rate", 60, 210, 150,
                                help="Maximum heart rate achieved during exercise test")
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0, 0.1,
                                help="ST depression induced by exercise relative to rest")
    
    # Test Results
    st.sidebar.markdown("### 🔬 Test Results")
    cp = st.sidebar.selectbox("Chest Pain Type", 
                              options=[(0, "Typical Angina"), (1, "Atypical Angina"), 
                                      (2, "Non-anginal Pain"), (3, "Asymptomatic")],
                              format_func=lambda x: x[1])
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                               options=[(0, "No"), (1, "Yes")],
                               format_func=lambda x: x[1])
    restecg = st.sidebar.selectbox("Resting ECG Results",
                                   options=[(0, "Normal"), (1, "ST-T Abnormality"), (2, "LV Hypertrophy")],
                                   format_func=lambda x: x[1])
    exang = st.sidebar.selectbox("Exercise Induced Angina",
                                 options=[(0, "No"), (1, "Yes")],
                                 format_func=lambda x: x[1])
    slope = st.sidebar.selectbox("ST Slope",
                                 options=[(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
                                 format_func=lambda x: x[1])
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)",
                              options=[0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia",
                                options=[(1, "Normal"), (2, "Fixed Defect"), (3, "Reversible Defect")],
                                format_func=lambda x: x[1])
    
    # Compile patient data
    patient_data = {
        'age': age,
        'sex': sex[1],
        'cp': cp[0],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs[0],
        'restecg': restecg[0],
        'thalach': thalach,
        'exang': exang[0],
        'oldpeak': oldpeak,
        'slope': slope[0],
        'ca': ca,
        'thal': thal[0]
    }
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Patient Summary")
        
        # Display patient info in a nice format
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown(f"""
            <div class="metric-card">
                <strong>👤 Demographics</strong><br>
                Age: {age} years<br>
                Sex: {sex[0]}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card" style="margin-top: 10px;">
                <strong>💓 Vital Signs</strong><br>
                BP: {trestbps} mm Hg<br>
                Max HR: {thalach} bpm
            </div>
            """, unsafe_allow_html=True)
        
        with info_col2:
            st.markdown(f"""
            <div class="metric-card">
                <strong>🧪 Lab Results</strong><br>
                Cholesterol: {chol} mg/dl<br>
                FBS > 120: {fbs[1]}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card" style="margin-top: 10px;">
                <strong>📈 ECG Results</strong><br>
                ST Depression: {oldpeak}<br>
                Chest Pain: {cp[1]}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Feature Comparison")
        radar_chart = create_feature_comparison_chart(patient_data)
        st.plotly_chart(radar_chart, use_container_width=True)
    
    # Prediction Button
    st.markdown("---")
    
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        predict_button = st.button("🔮 Predict Heart Disease Risk", use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner("Analyzing patient data..."):
            import time
            time.sleep(1)  # Simulate processing
            
            if demo_mode:
                # Demo prediction
                risk_score = (age/100 * 0.3 + trestbps/200 * 0.2 + chol/400 * 0.2 + 
                             (1 - thalach/200) * 0.15 + oldpeak/6 * 0.15)
                probability = [1 - risk_score, risk_score]
                prediction = 1 if risk_score > 0.5 else 0
            else:
                prediction, probability = predict_heart_disease(patient_data, model, fuzzy_transformer)
        
        # Display Results
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            # Risk Level Box
            risk_prob = probability[1]
            if risk_prob > 0.7:
                risk_level = "HIGH RISK"
                risk_class = "high-risk"
                recommendation = "⚠️ Immediate medical consultation recommended"
            elif risk_prob > 0.4:
                risk_level = "MEDIUM RISK"
                risk_class = "medium-risk"
                recommendation = "⚡ Schedule a follow-up with your doctor"
            else:
                risk_level = "LOW RISK"
                risk_class = "low-risk"
                recommendation = "✅ Continue healthy lifestyle practices"
            
            st.markdown(f"""
            <div class="prediction-box {risk_class}">
                {risk_level}<br>
                <span style="font-size: 2rem;">{risk_prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-box" style="margin-top: 15px;">
                <strong>📋 Recommendation:</strong><br>
                {recommendation}
            </div>
            """, unsafe_allow_html=True)
            
            # Probability breakdown
            st.markdown("### 📊 Probability Breakdown")
            prob_df = pd.DataFrame({
                'Outcome': ['No Heart Disease', 'Heart Disease'],
                'Probability': [probability[0] * 100, probability[1] * 100]
            })
            
            fig_bar = px.bar(prob_df, x='Outcome', y='Probability', 
                            color='Outcome',
                            color_discrete_map={'No Heart Disease': '#27ae60', 'Heart Disease': '#e74c3c'})
            fig_bar.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with result_col2:
            # Gauge Chart
            gauge_chart = create_gauge_chart(probability[1])
            st.plotly_chart(gauge_chart, use_container_width=True)
            
            # Risk Factors
            st.markdown("### ⚠️ Key Risk Factors")
            
            risk_factors = []
            if age > 55:
                risk_factors.append(("Age > 55", "age"))
            if trestbps > 140:
                risk_factors.append(("High Blood Pressure", "bp"))
            if chol > 240:
                risk_factors.append(("High Cholesterol", "chol"))
            if thalach < 120:
                risk_factors.append(("Low Max Heart Rate", "hr"))
            if oldpeak > 2:
                risk_factors.append(("Significant ST Depression", "st"))
            if exang[0] == 1:
                risk_factors.append(("Exercise-Induced Angina", "angina"))
            
            if risk_factors:
                for factor, _ in risk_factors:
                    st.markdown(f"🔴 {factor}")
            else:
                st.markdown("🟢 No major risk factors identified")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.<br>
        Always consult with a healthcare provider for medical decisions.<br><br>
        Built with ❤️ using Fuzzy Logic & LightGBM | © 2024
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
