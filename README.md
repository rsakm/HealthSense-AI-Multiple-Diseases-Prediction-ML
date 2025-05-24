# HealthSense AI: Multiple Disease Prediction System

![HealthSense AI Banner](https://healthsense-ai.streamlit.app/~/+/media/6a8f3e9c5a5a7e7e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation--setup)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Usage](#usage-guide)
- [Deployment](#deployment)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

HealthSense AI is a machine learning web application that predicts risks for:
- Diabetes Mellitus (92% accuracy)
- Cardiovascular Disease (88% accuracy) 
- Parkinson's Disease (90% accuracy)

**Live Demo**: [healthsense-ai.streamlit.app](https://healthsense-ai.streamlit.app)

## Features

### Predictive Capabilities
- Diabetes risk from clinical markers (glucose, BMI, etc.)
- Heart disease assessment via cardiovascular metrics
- Parkinson's detection through voice analysis

### User Experience
- Interactive input forms with validation
- Visual explanations of predictions
- Actionable health recommendations
- Responsive mobile-friendly design

### Technical Highlights
- Feature importance visualizations
- Probability-based confidence levels
- Model performance dashboards
- Clean modern UI with dark/light mode

## Technical Architecture

### Model Specifications

| Disease        | Algorithm             | Accuracy | Key Features           |
|----------------|-----------------------|----------|------------------------|
| Diabetes       | SVM (RBF Kernel)      | 92%      | Glucose, BMI, BP       |
| Heart Disease  | Logistic Regression   | 88%      | Cholesterol, ECG       |
| Parkinson's    | SVM (RBF Kernel)      | 90%      | Vocal jitter, HNR      |

### Tech Stack
- **Frontend**: Streamlit
- **Visualization**: Plotly, Seaborn
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Cloud

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
git clone https://github.com/rsakm/HealthSense-AI-Multiple-Diseases-Prediction-ML.git
cd HealthSense-AI-Multiple-Diseases-Prediction-ML
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── Notebooks/
│   ├── Diabetes_Prediction.ipynb
│   ├── Heart_Disease_Prediction.ipynb
│   └── Parkinsons_Prediction.ipynb
├── saved_models/
│   ├── diabetes_model.sav
│   ├── heart_model.sav
│   └── parkinsons_model.pkl
├── datasets/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
├── app.py
├── requirements.txt
└── .gitignore
```

## Model Training

1. **Data Preparation**
   - SMOTE for class imbalance
   - StandardScaler normalization
   - Feature engineering

2. **Model Development**
   - GridSearchCV hyperparameter tuning
   - 5-fold cross-validation
   - Permutation importance analysis

3. **Evaluation Metrics**
   - Precision, Recall, F1-Score
   - ROC-AUC curves
   - Confusion matrices

## Usage Guide

1. Select disease prediction tool from sidebar
2. Enter required health parameters
3. Click "Predict" button
4. View results with:
   - Risk probability percentage
   - Key contributing factors
   - Recommended actions

## Deployment

Deployed on Streamlit Cloud with automatic CI/CD from GitHub:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://healthsense-ai.streamlit.app)

## Future Roadmap

- [ ] Add user authentication
- [ ] Implement PDF report generation
- [ ] Expand to 10+ diseases
- [ ] Add time-series health tracking

## Contributing

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request  

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

**Project Maintainer**: RAJSHREE 
**Email**: rajshreek511@gmail.com 
**GitHub**: [rsakm](https://github.com/rsakm)  

[![GitHub Stars](https://img.shields.io/github/stars/rsakm/HealthSense-AI-Multiple-Diseases-Prediction-ML?style=social)](https://github.com/rsakm/HealthSense-AI-Multiple-Diseases-Prediction-ML/stargazers)  
[![GitHub Forks](https://img.shields.io/github/forks/rsakm/HealthSense-AI-Multiple-Diseases-Prediction-ML?style=social)](https:///github.com/rsakm/HealthSense-AI-Multiple-Diseases-Prediction-ML/network/members)
