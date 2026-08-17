# AgriPredict - AI-Driven Crop Yield Prediction

A Streamlit web application designed to empower Kenyan small-scale farmers and agricultural extension officers with real-time, AI-driven crop yield predictions and advanced analytics.

## Features

### Core Modules:
1. **Crop Yield Prediction**
   - Real-time yield predictions using ensemble ML models
   - Soil and weather data integration via APIs
   - Explainable AI with SHAP and feature importance
   - Agricultural recommendations

2. **Analytics Dashboard**
   - Historical prediction analysis
   - Interactive charts and visualizations
   - Export capabilities (Excel, PDF)
   - Performance tracking

3. **Help & Guidance**
   - Comprehensive user documentation
   - Troubleshooting guides
   - Agricultural resources

## Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: SQLite
- **ML Framework**: Scikit-learn, SHAP
- **Visualization**: Plotly, Altair
- **APIs**: OpenWeatherMap, SoilGrids

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tony405-spec/agriforecast.git
cd agriforecast
python -m pip install -r requirements.txt
streamlit run agripredict_app.py
```

## Generated Artifacts

The app may create local model and SQLite database files such as `trained_model.pkl` and `agripredict_ml.db`. These are runtime artifacts and should not be committed. Regenerate them locally by running the Streamlit app.
<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=22&color=FF073A&center=true&vCenter=true&width=1200&lines=🌱_INITIALIZING_AGRIPREDICT-KENYA_[████▓░░░░░]_40%_LOADING_DATA_PIPELINE...;🧠_LOADING_RANDOM_FOREST_%26_RESNET_ENSEMBLE_[██████▓░░]_65%_MODEL_WARMUP...;🌦️_INGESTING_WEATHER_%26_SOIL_FEATURES_[████████▓]_88%_FEATURE_ENGINEERING...;🔍_PREDICTING_CROP_YIELD_[██████████]_100%_ADVICE_READY_FOR_FARMERS...;🔗_OPEN_WEBAPP_%3A%20https%3A%2F%2Fagriprediction.streamlit.app%2F" alt="AgriPredict-Kenya status typing svg" />
</p>

<p align="center">
  <a href="https://agriforecaster.streamlit.app/" target="_blank">🔗 Open AgriPredict-Kenya web app</a>
</p>

