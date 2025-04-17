import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import os

# --- Load Model from Google Drive ---
@st.cache_resource
def load_model_from_gdrive(url, filename):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return joblib.load(filename)

# Google Drive URL for LightGBM model
lgbm_url = "https://drive.google.com/uc?export=download&id=18FFekDp6LfYUhdlncDgP-kcAh4MeqYZN"
lgbm_model = load_model_from_gdrive(rf_url, "best_rf_model.pkl")

# Simulate RFC Prediction
st.title("🧠 E-Commerce ML Prediction Dashboard")
st.markdown("---")
st.subheader("📥 Input Features for Prediction")

col1, col2 = st.columns(2)

with col1:
    user_views = st.number_input("User's Total Views", min_value=1, value=20)
    hour = st.slider("Hour of Day", 0, 23, 14)

with col2:
    item_views = st.number_input("Item's Total Views", min_value=1, value=50)
    price = st.number_input("Item Price", min_value=0.0, value=99.99, step=0.01)

if st.button("Predict with RFC Model"):
    X_sample = pd.DataFrame.from_dict({
        "user_total_views": [user_views],
        "item_total_views": [item_views],
        "hour": [hour],
        "price": [price]
    })

    with st.spinner("Predicting..."):
        prob = lgbm_model.predict_proba(X_sample)[:, 1][0]
        st.success(f"Predicted Purchase Probability: **{prob:.2%}**")

        if prob > 0.7:
            st.info("High conversion potential. Recommend premium placement.")
        elif prob > 0.4:
            st.info("Moderate conversion likelihood. Suggest promos or bundling.")
        else:
            st.info("Low conversion. Consider price/UX changes.")
