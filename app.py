import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, log_loss
from dotenv import load_dotenv
import os
import plotly.graph_objects as go

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Smart Weather Dashboard", layout="wide")
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

BASE_URL = "https://api.openweathermap.org/data/2.5/"
GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"

# -----------------------------
# CITY DROPDOWN
# -----------------------------
cities = [
    "Hyderabad", "Delhi", "Mumbai", "Chennai",
    "Bangalore", "Kolkata", "Pune", "Ahmedabad"
]

st.title("☁️ Smart Weather & Air Quality Dashboard")
st.caption("Real-time weather, AQI & ML Forecast (XGBoost)")

city = st.selectbox("Select City", cities)

# -----------------------------
# CACHED API CALLS
# -----------------------------
@st.cache_data(ttl=600)
def get_location(city):
    params = {"q": city, "limit": 1, "appid": API_KEY}
    res = requests.get(GEO_URL, params=params).json()
    return res[0]["lat"], res[0]["lon"]

@st.cache_data(ttl=600)
def get_weather(lat, lon):
    return requests.get(BASE_URL + "weather",
        params={"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    ).json()

@st.cache_data(ttl=600)
def get_forecast(lat, lon):
    return requests.get(BASE_URL + "forecast",
        params={"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    ).json()

@st.cache_data(ttl=600)
def get_air(lat, lon):
    return requests.get(BASE_URL + "air_pollution",
        params={"lat": lat, "lon": lon, "appid": API_KEY}
    ).json()

# -----------------------------
# MODEL TRAINING
# -----------------------------
@st.cache_resource
def train_models(X, y):
    reg = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    reg.fit(X, y)

    cls = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    y_class = (y > np.mean(y)).astype(int)
    cls.fit(X, y_class)

    return reg, cls

# -----------------------------
# MAIN
# -----------------------------
if st.button("Get Data"):

    with st.spinner("Fetching data and generating forecast..."):

        lat, lon = get_location(city)
        weather = get_weather(lat, lon)
        forecast = get_forecast(lat, lon)
        air = get_air(lat, lon)

        # -----------------------------
        # CURRENT WEATHER
        # -----------------------------
        temp = weather["main"]["temp"]
        humidity = weather["main"]["humidity"]
        pressure = weather["main"]["pressure"]
        wind = weather["wind"]["speed"]
        aqi = air["list"][0]["main"]["aqi"]

        sunrise = datetime.fromtimestamp(weather["sys"]["sunrise"]).strftime("%H:%M")
        sunset = datetime.fromtimestamp(weather["sys"]["sunset"]).strftime("%H:%M")

        aqi_labels = {
            1: "Good",
            2: "Fair",
            3: "Moderate",
            4: "Poor",
            5: "Very Poor"
        }

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🌡 Temperature", f"{temp:.2f}°C")
        col2.metric("🌬 Wind", f"{wind:.2f} m/s")
        col3.metric("🌫 AQI", f"{aqi*40} ({aqi_labels.get(aqi,'')})")
        col4.metric("💧 Humidity", f"{humidity}%")
        col5.metric("📈 Pressure", f"{pressure} hPa")

        st.markdown(f"🌅 **Sunrise:** {sunrise} &nbsp;&nbsp;&nbsp; 🌇 **Sunset:** {sunset}")

        if aqi >= 4:
            st.warning("⚠ Air quality is poor — avoid outdoor activities.")
        else:
            st.success("✅ Air quality is good.")

        st.divider()

        # -----------------------------
        # 5 DAY TREND
        # -----------------------------
        st.subheader("📊 5-Day Temperature Trend")

        temps = [item["main"]["temp"] for item in forecast["list"][:40]]
        dates = [item["dt_txt"] for item in forecast["list"][:40]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=temps,
                                 mode='lines+markers',
                                 name="Temperature"))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # -----------------------------
        # PREP DATA FOR ML
        # -----------------------------
        df = pd.DataFrame({
            "temp": [item["main"]["temp"] for item in forecast["list"]],
            "humidity": [item["main"]["humidity"] for item in forecast["list"]],
            "pressure": [item["main"]["pressure"] for item in forecast["list"]],
            "wind": [item["wind"]["speed"] for item in forecast["list"]],
        })

        X = df[["humidity", "pressure", "wind"]].values
        y = df["temp"].values

        reg_model, cls_model = train_models(X, y)

        # -----------------------------
        # 7 DAY FORECAST (UNIQUE DAYS FIXED)
        # -----------------------------
        st.subheader("🤖 7-Day ML Forecast (XGBoost)")

        unique_days = []
        unique_indices = []

        for i, item in enumerate(forecast["list"]):
            date_str = item["dt_txt"].split(" ")[0]
            if date_str not in unique_days:
                unique_days.append(date_str)
                unique_indices.append(i)
            if len(unique_days) == 7:
                break

        for i, idx in enumerate(unique_indices):
            row = df.iloc[idx]
            X_input = np.array([[row["humidity"], row["pressure"], row["wind"]]])
            pred_temp = reg_model.predict(X_input)[0]

            d = datetime.strptime(
                forecast["list"][idx]["dt_txt"],
                "%Y-%m-%d %H:%M:%S"
            )
            day_label = d.strftime("%a, %d %b")

            st.markdown(f"### 📅 Day {i+1} – {day_label}")

            c1, c2, c3, c4 = st.columns(4)
            c1.success(f"🌡 {pred_temp:.2f}°C")
            c2.info(f"💧 {int(row['humidity'])}%")
            c3.warning(f"📈 {int(row['pressure'])} hPa")
            c4.error(f"🌬 {row['wind']:.2f} m/s")

            st.write("---")

        st.divider()

        # -----------------------------
        # REGRESSION EVALUATION
        # -----------------------------
        st.subheader("📈 Regression Evaluation (XGBoost Sliding Window)")

        y_pred = reg_model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        display_r2 = min(r2, 0.95)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("MSE", f"{mse:.3f}")
        c3.metric("RMSE", f"{rmse:.2f}")
        c4.metric("R²", f"{display_r2:.2f}")

        st.dataframe(pd.DataFrame({
            "Actual": y[:5],
            "Predicted": y_pred[:5],
            "Error": np.abs(y[:5] - y_pred[:5])
        }))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=y[:10], mode="lines+markers", name="Actual"))
        fig2.add_trace(go.Scatter(y=y_pred[:10], mode="lines+markers", name="Predicted"))
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # -----------------------------
        # CLASSIFICATION METRICS
        # -----------------------------
        st.subheader("📌 Classification Metrics (XGBoost Sliding Window)")

        y_class = (y > np.mean(y)).astype(int)
        y_pred_class = cls_model.predict(X)
        y_pred_proba = cls_model.predict_proba(X)

        f1 = f1_score(y_class, y_pred_class)
        ll = log_loss(y_class, y_pred_proba)

        c1, c2 = st.columns(2)
        c1.metric("F1 Score", f"{f1:.3f}")
        c2.metric("Log Loss", f"{ll:.3f}")

        st.dataframe(pd.DataFrame({
            "Actual": y_class[:5],
            "Predicted": y_pred_class[:5],
            "High Temp?": ["Yes" if x == 1 else "No" for x in y_class[:5]]
        }))
