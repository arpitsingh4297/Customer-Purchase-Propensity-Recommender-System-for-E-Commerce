import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os
import requests
import io

# --- Google Drive File IDs ---
BEST_LGBM_MODEL_ID = "1YLqd99yXh8uqe2DhjV2IqhUvs9qGb5tF"
LABEL_ENCODER_ID = "1gqQhv8UwnGCZAPcnj9cO8zcG1rolpHgj"
SCALER_ID = "1J1oXrza060gwNsqfCQAHqyVg2s7eW2j0"

# --- Function to download file from Google Drive ---
@st.cache_resource
def download_file_from_gdrive(file_id, destination_name):
    """Downloads a file from Google Drive using its file ID."""
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(destination_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"✅ Successfully downloaded {destination_name}")
        return destination_name
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error downloading {destination_name} from Google Drive: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during download of {destination_name}: {e}")
        st.stop()

# --- Load model and encoders ---
@st.cache_resource
def load_model_artifacts():
    """Downloads and loads the model, label encoder, and scaler."""
    model_path = download_file_from_gdrive(BEST_LGBM_MODEL_ID, "best_lgbm_model.pkl")
    label_encoder_path = download_file_from_gdrive(LABEL_ENCODER_ID, "label_encoder.pkl")
    scaler_path = download_file_from_gdrive(SCALER_ID, "scaler.pkl")

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    scaler = joblib.load(scaler_path)
    return model, label_encoder, scaler

model, label_encoder, scaler = load_model_artifacts()

# --- Get the base directory of the script ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Sidebar Navigation ---
st.sidebar.title("🛒 E-Commerce Intelligence")
page = st.sidebar.radio("Navigation", ["Home", "Predict Purchase", "Insights & Explainability"])

# --- Home Page ---
if page == "Home":
    st.title("📦 Customer Purchase Propensity & Recommender System")
    st.markdown("""
    Welcome to the **E-Commerce Purchase Prediction App**!
    This tool predicts the likelihood of a customer purchasing an item and visualizes model performance.

    #### 🔍 Features:
    - Upload customer-item data to get predictions
    - Visual explainability (SHAP)
    - Recommender system metrics

    ---
    **Note:** Ensure your input CSV is preprocessed, encoded, and contains only numeric values.

    ---
    **To get started:**
    1. Go to the **Predict Purchase** tab
    2. Upload your preprocessed input CSV
    3. See predictions instantly!
    """)

# --- Predict Purchase ---
elif page == "Predict Purchase":
    st.title("🎯 Predict Customer Purchase Propensity")

    uploaded_file = st.file_uploader("📤 Upload Preprocessed Input File (CSV Format)", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("✅ Uploaded Data Preview:", input_df.head())

            # Validate shape with scaler
            expected_features = scaler.mean_.shape[0]
            if input_df.shape[1] != expected_features:
                st.error(f"Uploaded file has {input_df.shape[1]} features, but {expected_features} expected. Please check your input.")
                st.stop()

            # Preprocessing
            input_scaled = scaler.transform(input_df)

            # Predict
            predictions = model.predict(input_scaled)
            predictions_proba = model.predict_proba(input_scaled)[:, 1]

            input_df['Purchase_Prediction'] = predictions
            input_df['Purchase_Probability'] = np.round(predictions_proba, 3)

            # Only decode if label encoder was used on the target
            try:
                decoded = label_encoder.inverse_transform(predictions)
                input_df['Purchase_Label'] = decoded
            except:
                input_df['Purchase_Label'] = predictions

            st.success("✅ Prediction Successful!")
            st.dataframe(input_df)

            # Download
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name='predicted_purchase_propensity.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- Insights Page ---
elif page == "Insights & Explainability":
    st.title("📊 Model Insights & Explainability")

    # Recommender Bar Chart
    st.subheader("🔢 Recommender Model Metrics")
    try:
        recommender_img_path = os.path.join(BASE_DIR, "recommender_bar_metrics.png")
        recommender_img = Image.open(recommender_img_path)
        st.image(recommender_img, caption="Top-K Recommender Performance", use_column_width=True)
    except Exception as e:
        st.warning(f"Could not load recommender_bar_metrics.png: {e}")

    st.markdown("---")

    # SHAP Summary
    st.subheader("📌 SHAP Summary Plot - Feature Importance")
    try:
        shap_img_path = os.path.join(BASE_DIR, "shap_summary.png")
        shap_img = Image.open(shap_img_path)
        st.image(shap_img, caption="SHAP Summary Plot", use_column_width=True)
    except Exception as e:
        st.warning(f"Could not load shap_summary.png: {e}")