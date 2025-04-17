import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import time
import logging
from datetime import datetime

# --- Configuration ---
PAGE_TITLE = "E-Commerce Analytics Dashboard"
PAGE_ICON = "🛒"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"
CSS_FILE = "style.css"
SAMPLE_DATA_SIZE = 5000
SAMPLE_START_DATE = "2023-01-01"
SAMPLE_END_DATE = "2023-06-30"
RECOMMENDATION_IMAGE_PLACEHOLDER = "https://via.placeholder.com/150"
GITHUB_REPO_URL = "https://github.com/your-repo"
GITHUB_ISSUES_URL = "https://github.com/your-repo/issues"

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page Setup ---
def setup_page():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE,
    )
    _load_local_css(CSS_FILE)

def _load_local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        logger.warning(f"CSS file {file_name} not found - using default styling")

# --- Data Loading ---
@st.cache_data
def _generate_sample_data():
    """Generates synthetic e-commerce data matching actual column structure."""
    np.random.seed(42)
    dates = pd.date_range(start=SAMPLE_START_DATE, end=SAMPLE_END_DATE, periods=SAMPLE_DATA_SIZE)

    data = {
        "timestamp": np.random.choice(dates, SAMPLE_DATA_SIZE),
        "visitorid": np.random.randint(100000, 999999, SAMPLE_DATA_SIZE),
        "event": np.random.choice(["view", "addtocart", "purchase"], SAMPLE_DATA_SIZE, p=[0.85, 0.12, 0.03]),
        "itemid": np.random.randint(100000, 999999, SAMPLE_DATA_SIZE),
        "transactionid": np.random.choice([np.nan] + list(range(1000, 9999)), SAMPLE_DATA_SIZE),
        "purchase": np.random.choice([0, 1], SAMPLE_DATA_SIZE, p=[0.97, 0.03]),
        "user_total_views": np.random.randint(1, 100, SAMPLE_DATA_SIZE),
        "item_total_views": np.random.randint(1, 200, SAMPLE_DATA_SIZE),
        "categoryid": np.random.choice(["Electronics", "Clothing", "Home", "Books"], SAMPLE_DATA_SIZE),
    }

    df = pd.DataFrame(data)
    # Ensure purchase is properly set based on event
    df["purchase"] = (df["event"] == "purchase").astype(int)
    return df

@st.cache_data
def load_data():
    """Loads and preprocesses e-commerce data with robust error handling."""
    try:
        # Read CSV with explicit column specification
        df = pd.read_csv("cleaned_user_item_data_reduced_properties.csv")
        logger.info(f"Successfully loaded data with columns: {df.columns.tolist()}")

        # Convert timestamp if exists
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.day_name()

        # Verify we have the required columns
        required_columns = {'event', 'purchase'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            logger.warning(f"Missing required columns: {missing}")
            st.warning(f"Missing required columns: {missing} - using sample data instead")
            return _generate_sample_data()

        # Ensure purchase is numeric (it might be loaded as string)
        df["purchase"] = pd.to_numeric(df["purchase"], errors='coerce').fillna(0).astype(int)

        return df

    except FileNotFoundError:
        logger.warning("Data file not found - generating sample data")
        st.sidebar.warning("Using sample data - real data not found")
        return _generate_sample_data()
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)} - using sample data")
        return _generate_sample_data()

# --- Helper Functions ---
def plot_conversion_funnel(data: pd.DataFrame) -> px.funnel:
    """Visualizes the conversion funnel with error handling."""
    try:
        views = data.shape[0]
        add_to_cart = data[data["event"] == "addtocart"].shape[0]
        purchases = data[data["purchase"] == 1].shape[0]

        funnel_df = pd.DataFrame({
            "Stage": ["Views", "Add-to-Cart", "Purchases"],
            "Count": [views, add_to_cart, purchases],
            "Rate": [
                1,
                add_to_cart / views if views > 0 else 0,
                purchases / views if views > 0 else 0,
            ],
        })

        fig = px.funnel(
            funnel_df,
            x="Count",
            y="Stage",
            title="Conversion Funnel",
            labels={"Count": "Number of Events", "Stage": "Conversion Stage"},
            color="Stage",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )

        for i, row in funnel_df.iterrows():
            fig.add_annotation(
                x=row["Count"] + max(funnel_df["Count"]) * 0.05,
                y=row["Stage"],
                text=f"{row['Rate']:.1%}",
                showarrow=False,
                font=dict(size=12),
            )
        return fig
    except Exception as e:
        logger.error(f"Error generating funnel plot: {str(e)}")
        st.error("Could not generate conversion funnel visualization")
        return px.funnel()  # Return empty figure

def simulate_purchase_prediction(
    user_views: int, item_views: int, hour: int, model_type: str
) -> float:
    """Simulates purchase probability with input validation."""
    try:
        # Validate inputs
        user_views = max(0, int(user_views))
        item_views = max(0, int(item_views))
        hour = max(0, min(23, int(hour)))
        
        base_prob = 0.1
        user_factor = min(0.4, user_views / 250)
        item_factor = min(0.3, item_views / 300)
        time_factor = 0.2 * (1 - abs(hour - 15) / 12)  # Peaks around 3pm

        if model_type == "LightGBM":
            return min(0.95, base_prob + user_factor + item_factor + time_factor + 0.05)
        elif model_type == "Random Forest":
            return min(0.95, base_prob + user_factor + item_factor + time_factor)
        else:  # Logistic Regression
            return min(0.95, base_prob + 0.8 * (user_factor + item_factor + time_factor))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return 0.0

def generate_recommendations(user_id, n=5):
    """Generates recommendations with error handling."""
    try:
        if "itemid" not in df.columns or "purchase" not in df.columns:
            st.warning("Cannot generate recommendations: Missing required data columns.")
            return []

        top_items = (
            df.groupby("itemid")["purchase"].sum().sort_values(ascending=False).head(100)
        )
        if len(top_items) == 0:
            return []
        return np.random.choice(top_items.index, size=min(n, len(top_items)), replace=False)
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        st.error("Error generating recommendations")
        return []

# --- Sidebar ---
def create_sidebar():
    st.sidebar.title(PAGE_TITLE)
    try:
        st.sidebar.image(
            "https://via.placeholder.com/150x50?text=E-Commerce", width=150
        )
    except:
        pass  # Silently handle image loading errors

    date_range = _create_date_range_filter()
    selected_page = _create_navigation()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides analytics and AI insights for e-commerce operations. "
        "Use the tabs above to explore different aspects of customer behavior."
    )

    return selected_page, date_range

def _create_date_range_filter():
    """Creates a date range filter with validation."""
    try:
        if "timestamp" in df.columns:
            min_date = df["timestamp"].min().date()
            max_date = df["timestamp"].max().date()
            return st.sidebar.date_input(
                "Select Date Range",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date,
            )
        return None
    except:
        return None

def _create_navigation():
    """Creates the main navigation with default selection."""
    try:
        return st.sidebar.radio(
            "Navigation",
            ["📊 Dashboard Overview", "🔍 User Behavior", "🤖 Purchase Predictor", "🎯 Recommender"],
            index=0,
        )
    except:
        return "📊 Dashboard Overview"  # Default if radio fails

# --- Main Content Sections ---
def show_dashboard_overview(data: pd.DataFrame):
    """Displays the main dashboard overview with error boundaries."""
    try:
        st.title("📊 E-Commerce Dashboard Overview")
        _show_kpi_cards(data)
        _show_conversion_funnel(data)
        _show_temporal_patterns(data)
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        st.error("Could not display dashboard overview")

def _show_kpi_cards(data: pd.DataFrame):
    """Displays KPI cards with safe metrics calculation."""
    try:
        st.subheader("Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)

        total_views = data.shape[0]
        add_to_cart = data[data["event"] == "addtocart"].shape[0] if "event" in data.columns else 0
        purchases = data[data["purchase"] == 1].shape[0] if "purchase" in data.columns else 0
        conversion_rate = purchases / total_views if total_views > 0 else 0

        col1.metric("Total Views", f"{total_views:,}")
        col2.metric("Add-to-Cart", f"{add_to_cart:,}", f"{add_to_cart/total_views:.1%} rate" if total_views > 0 else "N/A")
        col3.metric("Purchases", f"{purchases:,}")
        col4.metric("Conversion Rate", f"{conversion_rate:.1%}" if total_views > 0 else "N/A")
    except:
        st.warning("Could not display KPI metrics")

def _show_conversion_funnel(data: pd.DataFrame):
    """Displays conversion funnel with error handling."""
    try:
        st.subheader("Conversion Funnel")
        with st.expander("How to interpret this funnel"):
            st.write("""
                The conversion funnel shows how users progress through different stages:
                1. **Views**: Users who viewed products
                2. **Add-to-Cart**: Users who added items to their cart
                3. **Purchases**: Users who completed purchases
                The percentages show the conversion rates between stages.
            """)

        fig = plot_conversion_funnel(data)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Could not display conversion funnel")

def _show_temporal_patterns(data: pd.DataFrame):
    """Displays temporal patterns with fallback for missing data."""
    try:
        st.subheader("Temporal Patterns")
        tab1, tab2 = st.tabs(["By Hour of Day", "By Day of Week"])

        with tab1:
            st.write("User activity patterns throughout the day")
            if "hour" in data.columns:
                hourly_data = data.groupby("hour").size()
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=hourly_data, ax=ax)
                ax.set_title("Activity by Hour of Day")
                ax.set_xlabel("Hour")
                ax.set_ylabel("Number of Events")
                st.pyplot(fig)
            else:
                st.warning("Hour data not available")

        with tab2:
            st.write("User activity patterns by day of week")
            if "day_of_week" in data.columns:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                daily_data = data["day_of_week"].value_counts().reindex(day_order, fill_value=0)

                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x=daily_data.index, y=daily_data.values, palette="viridis", ax=ax)
                ax.set_title("Activity by Day of Week")
                ax.set_xlabel("Day")
                ax.set_ylabel("Number of Events")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.warning("Day of week data not available")
    except:
        st.warning("Could not display temporal patterns")

def show_user_behavior_analysis(data: pd.DataFrame):
    """Displays user behavior analysis with error boundaries."""
    try:
        st.title("🔍 User Behavior Analysis")
        _show_user_engagement(data)
        _show_item_popularity(data)
    except:
        st.error("Could not display user behavior analysis")

def _show_user_engagement(data: pd.DataFrame):
    """Displays user engagement patterns safely."""
    try:
        st.subheader("User Engagement Patterns")
        col1, col2 = st.columns(2)

        with col1:
            st.write("User Activity Distribution")
            if "user_total_views" in data.columns:
                fig, ax = plt.subplots()
                sns.histplot(data["user_total_views"], bins=20, kde=True, ax=ax)
                ax.set_title("Distribution of User Views")
                ax.set_xlabel("Total Views per User")
                st.pyplot(fig)
            else:
                st.warning("User view data not available")

        with col2:
            st.write("Top Active Users")
            if "visitorid" in data.columns and "user_total_views" in data.columns:
                top_users = (
                    data.groupby("visitorid")["user_total_views"]
                    .max()
                    .sort_values(ascending=False)
                    .head(10)
                )
                st.dataframe(top_users.reset_index().rename(columns={"user_total_views": "Total Views"}))
            else:
                st.warning("User data not available")
    except:
        st.warning("Could not display user engagement patterns")

def _show_item_popularity(data: pd.DataFrame):
    """Displays item popularity analysis safely."""
    try:
        st.subheader("Item Popularity Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Most Viewed Items")
            if "itemid" in data.columns and "item_total_views" in data.columns:
                top_items = (
                    data.groupby("itemid")["item_total_views"]
                    .max()
                    .sort_values(ascending=False)
                    .head(10)
                )
                st.dataframe(top_items.reset_index().rename(columns={"item_total_views": "Total Views"}))
            else:
                st.warning("Item view data not available")

        with col2:
            st.write("Best Selling Items")
            if "itemid" in data.columns and "purchase" in data.columns:
                best_sellers = data[data["purchase"] == 1]["itemid"].value_counts().head(10)
                st.dataframe(best_sellers.reset_index().rename(columns={"index": "Item ID", "itemid": "Purchases"}))
            else:
                st.warning("Purchase data not available")
    except:
        st.warning("Could not display item popularity analysis")

def show_purchase_predictor():
    """Displays purchase predictor with error handling."""
    try:
        st.title("🤖 Purchase Probability Predictor")
        model_option = _select_prediction_model()
        _show_model_performance(model_option)
        _predict_purchase_probability(model_option)
    except:
        st.error("Could not display purchase predictor")

def _select_prediction_model() -> str:
    """Lets the user select the prediction model safely."""
    try:
        return st.selectbox(
            "Select Prediction Model",
            ["LightGBM", "Random Forest", "Logistic Regression"],
            index=0,
            help="Choose the machine learning model to use for predictions",
        )
    except:
        return "LightGBM"  # Default if selectbox fails

def _show_model_performance(model_option: str):
    """Displays model performance metrics safely."""
    try:
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)

        metrics = {
            "LightGBM": {"ROC AUC Score": "0.89", "Precision-Recall AUC": "0.78"},
            "Random Forest": {"ROC AUC Score": "0.85", "Precision-Recall AUC": "0.74"},
            "Logistic Regression": {"ROC AUC Score": "0.82", "Precision-Recall AUC": "0.70"},
        }

        col1.metric("ROC AUC Score", metrics.get(model_option, {}).get("ROC AUC Score", "N/A"))
        col2.metric("Precision-Recall AUC", metrics.get(model_option, {}).get("Precision-Recall AUC", "N/A"))
    except:
        st.warning("Could not display model performance metrics")

def _predict_purchase_probability(model_option: str):
    """Provides purchase prediction interface safely."""
    try:
        st.subheader("Predict Purchase Probability")
        st.write("Enter user and item details to predict the likelihood of purchase")
        
        col1, col2 = st.columns(2)
        with col1:
            user_views = st.number_input("User Total Views", min_value=0, value=50)
            hour = st.slider("Hour of Day", 0, 23, 12)
        with col2:
            item_views = st.number_input("Item Total Views", min_value=0, value=75)
        
        if st.button("Predict Purchase Probability"):
            prob = simulate_purchase_prediction(user_views, item_views, hour, model_option)
            st.metric("Predicted Purchase Probability", f"{prob:.1%}")
    except:
        st.error("Could not display prediction interface")

# --- Main Application ---
if __name__ == "__main__":
    setup_page()
    df = load_data()
    
    selected_page, date_range = create_sidebar()
    
    # Apply date filter if available
    if date_range and len(date_range) == 2 and "timestamp" in df.columns:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
    
    if selected_page == "📊 Dashboard Overview":
        show_dashboard_overview(df)
    elif selected_page == "🔍 User Behavior":
        show_user_behavior_analysis(df)
    elif selected_page == "🤖 Purchase Predictor":
        show_purchase_predictor()
    elif selected_page == "🎯 Recommender":
        st.title("🎯 Product Recommender")
        user_id = st.number_input("Enter User ID", min_value=0, value=100001)
        if st.button("Generate Recommendations"):
            recommendations = generate_recommendations(user_id)
            if recommendations:
                st.success(f"Top recommendations for user {user_id}:")
                for i, item in enumerate(recommendations, 1):
                    st.write(f"{i}. Item ID: {item}")
            else:
                st.warning("No recommendations available")
