# 🫀 Intelligent Cardiac Diagnosis System

A hybrid intelligent system for cardiac disease diagnosis using **Fuzzy Logic** and **LightGBM**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Local Deployment](#local-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [License](#license)

---

## 🎯 Overview

This project implements a hybrid approach combining:
- **Fuzzy Logic**: For handling uncertainty in medical data
- **LightGBM**: For high-accuracy classification

The system transforms clinical measurements into fuzzy membership degrees, providing both accurate predictions and interpretable results.

---

## ✨ Features

- 🎯 **Accurate Predictions**: High accuracy using hybrid Fuzzy-LightGBM approach
- 📊 **Interactive Dashboard**: Beautiful Streamlit web interface
- 📈 **Visual Analytics**: Gauge charts, radar plots, and risk visualization
- 🔍 **Interpretable Results**: Fuzzy logic provides explainable features
- ☁️ **Cloud Ready**: Easy deployment to Streamlit Cloud (FREE)

---

## 📁 Project Structure

```
cardiac_deployment/
├── app.py                    # Main Streamlit application
├── fuzzy_transformer.py      # Fuzzy logic feature transformer
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── model/
│   └── hybrid_cardiac_model.pkl  # Trained model (after training)
└── .streamlit/
    └── config.toml           # Streamlit configuration (optional)
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download the Project

```bash
# If using git
git clone <your-repo-url>
cd cardiac_deployment

# Or simply download and extract the files
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥️ Local Deployment

### Step 1: Train the Model

```bash
python train_model.py
```

This will:
- Download the UCI Heart Disease dataset
- Apply fuzzy transformations
- Train the LightGBM model
- Save to `model/hybrid_cardiac_model.pkl`

### Step 2: Run the Application

```bash
streamlit run app.py
```

### Step 3: Access the App

Open your browser and go to:
```
http://localhost:8501
```

---

## ☁️ Cloud Deployment (Streamlit Cloud - FREE!)

### Step 1: Push to GitHub

1. Create a new GitHub repository
2. Push all files to the repository:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cardiac-diagnosis.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set the main file path: `app.py`
6. Click **"Deploy"**

### Step 3: Your App is Live! 🎉

Your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

---

## 📱 Alternative Deployment Options

### Option A: Heroku

1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Option B: Railway

1. Connect your GitHub repo to [Railway](https://railway.app)
2. Add start command: `streamlit run app.py`
3. Deploy automatically

### Option C: Docker

1. Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Train model
RUN python train_model.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run:
```bash
docker build -t cardiac-diagnosis .
docker run -p 8501:8501 cardiac-diagnosis
```

---

## 🎮 Usage

### Input Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| Age | Patient's age | 20-90 years |
| Sex | Gender | Male/Female |
| Chest Pain | Type of chest pain | 4 types |
| Blood Pressure | Resting BP | 80-200 mm Hg |
| Cholesterol | Serum cholesterol | 100-400 mg/dl |
| Blood Sugar | Fasting BS > 120 | Yes/No |
| ECG | Resting ECG results | 3 types |
| Max Heart Rate | Maximum HR achieved | 60-210 bpm |
| Exercise Angina | Exercise-induced angina | Yes/No |
| ST Depression | ST depression value | 0-6 |
| ST Slope | Slope of ST segment | 3 types |
| Major Vessels | Number (0-3) | 0-3 |
| Thalassemia | Blood disorder type | 3 types |

### Output

- **Risk Level**: HIGH / MEDIUM / LOW
- **Probability**: Percentage likelihood of heart disease
- **Recommendations**: Clinical guidance based on risk
- **Visual Charts**: Gauge chart and radar plot

---

## 📸 Screenshots

### Main Interface
```
┌─────────────────────────────────────────────┐
│  🫀 Intelligent Cardiac Diagnosis System    │
│  ─────────────────────────────────────────  │
│                                             │
│  📋 Patient Summary    │    📈 Comparison   │
│  ┌─────────────────┐   │   [Radar Chart]    │
│  │ Age: 55         │   │                    │
│  │ BP: 140 mm Hg   │   │                    │
│  │ Chol: 250 mg/dl │   │                    │
│  └─────────────────┘   │                    │
│                                             │
│         [🔮 Predict Heart Disease Risk]     │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │     MEDIUM RISK - 58.3%             │   │
│  │     [Gauge Chart Visualization]      │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## 🛠️ Troubleshooting

### Model not found error
```bash
# Re-train the model
python train_model.py
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Port already in use
```bash
# Use a different port
streamlit run app.py --server.port=8502
```

---

## 📚 Technical Details

### Fuzzy Features (17 total)

| Original Feature | Fuzzy Variables |
|-----------------|-----------------|
| Age | young, middle, old |
| Blood Pressure | low, normal, high, very_high |
| Cholesterol | low, normal, high, very_high |
| Heart Rate | low, normal, high |
| ST Depression | low, medium, high |

### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~85% |
| F1-Score | ~84% |
| AUC-ROC | ~90% |

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and should not replace professional medical advice. Always consult with a healthcare provider for medical decisions.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Streamlit for the amazing web framework
- scikit-fuzzy for fuzzy logic implementation

---

**Built with ❤️ for Academic Purpose**
