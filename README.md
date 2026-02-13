# ☁️ Smart Weather & Air Quality Dashboard  

Real-time Weather Intelligence & ML Forecasting System built with Streamlit + XGBoost

---

##  Project Overview

This project is an end-to-end data application that combines:

-  Real-time weather data
-  Air Quality Index (AQI) with severity labels
-  Sunrise & Sunset insights
-  Interactive 5-day temperature trend visualization
-  7-Day Machine Learning temperature forecasting
-  Regression & Classification model evaluation

It demonstrates practical integration of APIs, data preprocessing, machine learning, performance evaluation, and production-style UI deployment.

---

##  Machine Learning Architecture

**Regression Model**
- XGBoost Regressor
- Sliding window training approach
- Metrics: MAE, MSE, RMSE, R²

**Classification Model**
- XGBoost Classifier
- Predicts high-temperature days
- Metrics: F1 Score, Log Loss

Models are cached for performance optimization and fast re-runs.

---

##  Tech Stack

- **Frontend & Deployment:** Streamlit  
- **Machine Learning:** XGBoost  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Plotly  
- **API Integration:** OpenWeather API  
- **Environment Management:** python-dotenv  

---

##  Performance & Optimization

- API response caching (`st.cache_data`)
- Model caching (`st.cache_resource`)
- Clean UI with metric cards & responsive layout
- Environment-based API key management

---

##  Installation

Clone the repository:

```
git clone https://github.com/your-username/weather-prediction-system.git
cd weather-prediction-system
```

Install dependencies:

```
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENWEATHER_API_KEY=your_api_key_here
```

Run the application:

```
streamlit run app.py
```

---

##  Conclusion

This project demonstrates how real-time API data can be integrated with machine learning models to build a practical forecasting system.  

By combining data preprocessing, model training (XGBoost), evaluation metrics, and an interactive Streamlit interface, the application delivers both analytical insights and user-friendly visualization in a production-style setup.

It reflects an end-to-end ML workflow — from data ingestion to deployment.


