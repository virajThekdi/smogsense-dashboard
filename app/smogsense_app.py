import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SmogSense", layout="wide")
st.title("🌫️ SmogSense: AI Air Quality Forecast")

# 🔁 Load trained model
try:
    model = joblib.load("models/aqi_forecast_model.pkl")
except FileNotFoundError:
    st.error("❌ Trained model not found. Please run 'model_training.ipynb' first.")
    st.stop()

# 📤 Upload AQI CSV
uploaded = st.file_uploader("📂 Upload AQI CSV File", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek

    # Predict AQI
    X = df[['hour', 'dayofweek']]
    df['Predicted_AQI'] = model.predict(X)

    # 📈 Show forecast chart
    st.subheader("📊 Predicted AQI Over Time")
    st.line_chart(df.set_index('datetime')['Predicted_AQI'])

    # 🚨 Alerts
    st.subheader("⚠️ Pollution Risk Alerts")
    peak = df['Predicted_AQI'].max()
    if peak > 200:
        st.error("❗ Very High Pollution Expected. Stay indoors and wear a mask.")
    elif peak > 100:
        st.warning("⚠️ Moderate Pollution Expected. Consider limiting outdoor exposure.")
    else:
        st.success("✅ Air Quality is Acceptable.")

    # 🧾 Show table
    st.subheader("🔍 Data Preview")
    st.dataframe(df[['datetime', 'AQI', 'Predicted_AQI']].head(20))

else:
    st.info("👆 Upload a CSV file with 'datetime' and 'AQI' columns to begin.")
