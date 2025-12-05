"""
🫀 Intelligent Cardiac Diagnosis System
A Hybrid Approach Using Fuzzy Logic & LightGBM

Streamlit Web Application - Cloud Ready (Self-Contained)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #ebf5fb;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)


# ============ FUZZY FEATURE TRANSFORMER (BUILT-IN) ============
import skfuzzy as fuzz

class FuzzyFeatureTransformer:
    """Transforms raw clinical features into fuzzy membership degrees."""
    
    def __init__(self):
        self._define_membership_functions()
    
    def _define_membership_functions(self):
        # Age
        self.age_range = np.arange(0, 101, 1)
        self.age_young = fuzz.trapmf(self.age_range, [0, 0, 30, 40])
        self.age_middle = fuzz.trapmf(self.age_range, [35, 45, 55, 65])
        self.age_old = fuzz.trapmf(self.age_range, [55, 65, 100, 100])
        
        # Blood Pressure
        self.bp_range = np.arange(80, 210, 1)
        self.bp_low = fuzz.trapmf(self.bp_range, [80, 80, 90, 110])
        self.bp_normal = fuzz.trapmf(self.bp_range, [100, 110, 120, 130])
        self.bp_high = fuzz.trapmf(self.bp_range, [125, 135, 145, 155])
        self.bp_very_high = fuzz.trapmf(self.bp_range, [150, 165, 210, 210])
        
        # Cholesterol
        self.chol_range = np.arange(100, 600, 1)
        self.chol_low = fuzz.trapmf(self.chol_range, [100, 100, 150, 180])
        self.chol_normal = fuzz.trapmf(self.chol_range, [170, 190, 210, 230])
        self.chol_high = fuzz.trapmf(self.chol_range, [220, 250, 280, 310])
        self.chol_very_high = fuzz.trapmf(self.chol_range, [300, 350, 600, 600])
        
        # Heart Rate
        self.hr_range = np.arange(60, 220, 1)
        self.hr_low = fuzz.trapmf(self.hr_range, [60, 60, 100, 120])
        self.hr_normal = fuzz.trapmf(self.hr_range, [110, 130, 160, 175])
        self.hr_high = fuzz.trapmf(self.hr_range, [165, 180, 220, 220])
        
        # ST Depression
        self.st_range = np.arange(0, 7, 0.1)
        self.st_low = fuzz.trapmf(self.st_range, [0, 0, 0.5, 1.0])
        self.st_medium = fuzz.trapmf(self.st_range, [0.8, 1.5, 2.0, 2.5])
        self.st_high = fuzz.trapmf(self.st_range, [2.0, 3.0, 7, 7])
    
    def get_membership_degree(self, value, universe, membership_function):
        return fuzz.interp_membership(universe, membership_function, value)
    
    def transform_single(self, row):
        fuzzy_features = {}
        
        # Age
        age = row['age']
        fuzzy_features['age_young'] = self.get_membership_degree(age, self.age_range, self.age_young)
        fuzzy_features['age_middle'] = self.get_membership_degree(age, self.age_range, self.age_middle)
        fuzzy_features['age_old'] = self.get_membership_degree(age, self.age_range, self.age_old)
        
        # Blood Pressure
        bp = row['trestbps']
        fuzzy_features['bp_low'] = self.get_membership_degree(bp, self.bp_range, self.bp_low)
        fuzzy_features['bp_normal'] = self.get_membership_degree(bp, self.bp_range, self.bp_normal)
        fuzzy_features['bp_high'] = self.get_membership_degree(bp, self.bp_range, self.bp_high)
        fuzzy_features['bp_very_high'] = self.get_membership_degree(bp, self.bp_range, self.bp_very_high)
        
        # Cholesterol
        chol = row['chol']
        fuzzy_features['chol_low'] = self.get_membership_degree(chol, self.chol_range, self.chol_low)
        fuzzy_features['chol_normal'] = self.get_membership_degree(chol, self.chol_range, self.chol_normal)
        fuzzy_features['chol_high'] = self.get_membership_degree(chol, self.chol_range, self.chol_high)
        fuzzy_features['chol_very_high'] = self.get_membership_degree(chol, self.chol_range, self.chol_very_high)
        
        # Heart Rate
        hr = row['thalach']
        fuzzy_features['hr_low'] = self.get_membership_degree(hr, self.hr_range, self.hr_low)
        fuzzy_features['hr_normal'] = self.get_membership_degree(hr, self.hr_range, self.hr_normal)
        fuzzy_features['hr_high'] = self.get_membership_degree(hr, self.hr_range, self.hr_high)
        
        # ST Depression
        st = row['oldpeak']
        fuzzy_features['st_low'] = self.get_membership_degree(st, self.st_range, self.st_low)
        fuzzy_features['st_medium'] = self.get_membership_degree(st, self.st_range, self.st_medium)
        fuzzy_features['st_high'] = self.get_membership_degree(st, self.st_range, self.st_high)
        
        return fuzzy_features
    
    def transform(self, df):
        return df.apply(lambda row: pd.Series(self.transform_single(row)), axis=1)


# ============ MODEL TRAINING FUNCTION ============
@st.cache_resource
def train_and_load_model():
    """Train the model on first run and cache it."""
    
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    
    # Load data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv(url, names=column_names, na_values='?')
    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int)
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
    df = df.dropna()
    
    # Fuzzy transformation
    fuzzy_transformer = FuzzyFeatureTransformer()
    fuzzy_features_df = fuzzy_transformer.transform(df)
    
    # Prepare features
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    X_original = df.drop('target', axis=1)
    X_combined = pd.concat([X_original[categorical_features], fuzzy_features_df], axis=1)
    y = df['target']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 200,
        'class_weight': 'balanced'
    }
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train)
    
    return model, fuzzy_transformer, list(X_combined.columns)


# ============ PREDICTION FUNCTION ============
def predict_heart_disease(patient_data, model, fuzzy_transformer):
    """Make prediction for a patient."""
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    patient_df = pd.DataFrame([patient_data])
    fuzzy_features = fuzzy_transformer.transform(patient_df)
    X = pd.concat([patient_df[categorical_features], fuzzy_features], axis=1)
    
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
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_feature_comparison_chart(patient_data):
    """Create a radar chart comparing patient features to normal ranges."""
    categories = ['Age', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'ST Depression']
    
    patient_values = [
        (patient_data['age'] - 20) / (80 - 20) * 100,
        (patient_data['trestbps'] - 80) / (200 - 80) * 100,
        (patient_data['chol'] - 100) / (400 - 100) * 100,
        (patient_data['thalach'] - 60) / (200 - 60) * 100,
        patient_data['oldpeak'] / 6 * 100
    ]
    
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
    
    # Load/Train model
    with st.spinner("🔄 Loading AI Model (first time may take a minute)..."):
        model, fuzzy_transformer, feature_names = train_and_load_model()
    
    # Sidebar - Patient Information Input
    st.sidebar.markdown("## 📋 Patient Information")
    st.sidebar.markdown("---")
    
    # Personal Information
    st.sidebar.markdown("### 👤 Personal Details")
    age = st.sidebar.slider("Age (years)", 20, 90, 55, help="Patient's age in years")
    sex = st.sidebar.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
    
    # Clinical Measurements
    st.sidebar.markdown("### 🏥 Clinical Measurements")
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 240)
    thalach = st.sidebar.slider("Max Heart Rate", 60, 210, 150)
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
    
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
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia",
                                options=[(1, "Normal"), (2, "Fixed Defect"), (3, "Reversible Defect")],
                                format_func=lambda x: x[1])
    
    # Compile patient data
    patient_data = {
        'age': age, 'sex': sex[1], 'cp': cp[0], 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs[0], 'restecg': restecg[0], 'thalach': thalach,
        'exang': exang[0], 'oldpeak': oldpeak, 'slope': slope[0], 'ca': ca, 'thal': thal[0]
    }
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Patient Summary")
        
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
            <div class="metric-card">
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
            <div class="metric-card">
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
        with st.spinner("🔬 Analyzing patient data with Fuzzy Logic + LightGBM..."):
            import time
            time.sleep(0.5)
            prediction, probability = predict_heart_disease(patient_data, model, fuzzy_transformer)
        
        # Display Results
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
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
            gauge_chart = create_gauge_chart(probability[1])
            st.plotly_chart(gauge_chart, use_container_width=True)
            
            st.markdown("### ⚠️ Key Risk Factors")
            
            risk_factors = []
            if age > 55:
                risk_factors.append("🔴 Age > 55")
            if trestbps > 140:
                risk_factors.append("🔴 High Blood Pressure")
            if chol > 240:
                risk_factors.append("🔴 High Cholesterol")
            if thalach < 120:
                risk_factors.append("🔴 Low Max Heart Rate")
            if oldpeak > 2:
                risk_factors.append("🔴 Significant ST Depression")
            if exang[0] == 1:
                risk_factors.append("🔴 Exercise-Induced Angina")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(factor)
            else:
                st.markdown("🟢 No major risk factors identified")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.<br>
        Always consult with a healthcare provider for medical decisions.<br><br>
        Built with ❤️ using Fuzzy Logic & LightGBM
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
