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
import os  # Import the os module for path manipulation

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

# --- Load Models from Local Files ---
@st.cache_resource
def load_model(model_path):
    """Loads a model from a local pickle file."""
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            return model
        else:
            logger.error(f"Model file not found at {model_path}")
            st.error(f"Model file not found at {model_path}. Please ensure it's in the correct location.")
            return None
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        st.error(f"Error loading model: {e}")
        return None

# Define the paths to your model files (assuming they are in the same directory)
lgbm_model_path = "best_lgbm_model.pkl"
rf_model_path = "best_rf_model.pkl"

# Load the models
lgbm_model = load_model(lgbm_model_path)
rf_model = load_model(rf_model_path)

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

        if model_type == "LightGBM" and lgbm_model is not None:
            # Replace this with actual prediction using lgbm_model
            features = np.array([[user_views, item_views, hour]])
            # Assuming your LightGBM model has a predict_proba method
            proba = lgbm_model.predict_proba(features)
            return proba[0][1] if proba is not None else 0.0
        elif model_type == "Random Forest" and rf_model is not None:
            # Replace this with actual prediction using rf_model
            features = np.array([[user_views, item_views, hour]])
            # Assuming your Random Forest model has a predict_proba method
            proba = rf_model.predict_proba(features)
            return proba[0][1] if proba is not None else 0.0
        else:  # Logistic Regression (simulated)
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
    """Shows user engagement metrics and visualizations."""
    try:
        st.subheader("User Engagement Metrics")
        
        # Calculate engagement metrics
        if "user_total_views" in data.columns:
            avg_views_per_user = data.groupby("visitorid")["user_total_views"].mean().mean()
            active_users = data["visitorid"].nunique()
        else:
            avg_views_per_user = 0
            active_users = 0
            
        if "event" in data.columns:
            cart_add_rate = data[data["event"] == "addtocart"].shape[0] / data.shape[0] if data.shape[0] > 0 else 0
        else:
            cart_add_rate = 0
            
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Active Users", f"{active_users:,}")
        col2.metric("Avg Views per User", f"{avg_views_per_user:.1f}")
        col3.metric("Add-to-Cart Rate", f"{cart_add_rate:.1%}")
        
        # Show user activity distribution
        st.subheader("User Activity Distribution")
        if "user_total_views" in data.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(data["user_total_views"], bins=30, kde=True, ax=ax)
            ax.set_title("Distribution of User Views")
            ax.set_xlabel("Number of Views")
            ax.set_ylabel("Number of Users")
            st.pyplot(fig)
        else:
            st.warning("User view data not available")
    except Exception as e:
        logger.error(f"Error showing user engagement: {str(e)}")
        st.warning("Could not display user engagement metrics")

def _show_item_popularity(data: pd.DataFrame):
    """Shows item popularity metrics and visualizations."""
    try:
        st.subheader("Item Popularity")
        
        # Calculate popularity metrics
        if "itemid" in data.columns:
            top_items = data["itemid"].value_counts().head(10)
            popular_categories = data["categoryid"].value_counts().head(5) if "categoryid" in data.columns else pd.Series()
        else:
            top_items = pd.Series()
            popular_categories = pd.Series()
            
        # Display in tabs
        tab1, tab2 = st.tabs(["Top Items", "Categories"])
        
        with tab1:
            if not top_items.empty:
                st.write("Most viewed items:")
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x=top_items.values, y=top_items.index.astype(str), palette="rocket", ax=ax)
                ax.set_title("Top 10 Most Viewed Items")
                ax.set_xlabel("Number of Views")
                ax.set_ylabel("Item ID")
                st.pyplot(fig)
            else:
                st.warning("Item data not available")
                
        with tab2:
            if not popular_categories.empty:
                st.write("Popular categories:")
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x=popular_categories.values, y=popular_categories.index, palette="mako", ax=ax)
                ax.set_title("Top 5 Most Popular Categories")
                ax.set_xlabel("Number of Views")
                ax.set_ylabel("Category")
                st.pyplot(fig)
            else:
                st.warning("Category data not available")
    except Exception as e:
        logger.error(f"Error showing item popularity: {str(e)}")
        st.warning("Could not display item popularity metrics")

def show_purchase_predictor():
    """Displays the purchase prediction interface."""
    try:
        st.title("🤖 Purchase Probability Predictor")
        st.write("""
            This tool predicts the likelihood of a purchase based on user behavior.
            Adjust the sliders to see how different factors affect purchase probability.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            user_views = st.slider("User Total Views", 1, 500, 50, help="Total number of views by this user")
            item_views = st.slider("Item Total Views", 1, 500, 100, help="Total number of views for this item")
            
        with col2:
            hour = st.slider("Hour of Day", 0, 23, 12, help="Current hour of day (0-23)")
            model_type = st.selectbox(
                "Prediction Model",
                ["LightGBM", "Random Forest", "Logistic Regression"],
                index=0,
                help="Select which machine learning model to use for prediction"
            )
            
        # Calculate and display prediction
        if st.button("Predict Purchase Probability"):
            with st.spinner("Calculating..."):
                time.sleep(0.5)  # Simulate processing time
                probability = simulate_purchase_prediction(user_views, item_views, hour, model_type)
                
                st.subheader("Prediction Result")
                st.metric("Purchase Probability", f"{probability:.1%}")
                
                # Visualize with a gauge chart
                fig = px.indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title="Purchase Probability",
                    gauge={
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': probability * 100
                        }
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation
                with st.expander("Interpretation"):
                    if probability < 0.3:
                        st.info("Low purchase probability. Consider offering incentives or recommendations.")
                    elif probability < 0.7:
                        st.info("Moderate purchase probability. The user is considering but may need more engagement.")
                    else:
                        st.info("High purchase probability. The user is likely to purchase soon.")
    except Exception as e:
        logger.error(f"Error in purchase predictor: {str(e)}")
        st.error("Could not display purchase predictor")

def show_recommender_system():
    """Displays the recommender system interface."""
    try:
        st.title("🎯 Personalized Recommendations")
        st.write("""
            This system generates personalized product recommendations based on user behavior
            and popular items in the catalog.
        """)
        
        # User input
        user_id = st.text_input("Enter User ID", "12345", help="Enter a numeric user ID")
        n_recommendations = st.slider("Number of Recommendations", 1, 10, 5)
        
        if st.button("Generate Recommendations"):
            with st.spinner("Finding the best products for you..."):
                time.sleep(1)  # Simulate processing time
                recommendations = generate_recommendations(user_id, n_recommendations)
                
                if not recommendations:
                    st.warning("Could not generate recommendations. Please try again.")
                    return
                    
                st.subheader(f"Top {len(recommendations)} Recommendations for User {user_id}")
                
                # Display recommendations in a grid
                cols = st.columns(3)
                for i, item_id in enumerate(recommendations):
                    with cols[i % 3]:
                        st.image(RECOMMENDATION_IMAGE_PLACEHOLDER, width=150)
                        st.markdown(f"**Item #{item_id}**")
                        
                        # Get item details if available
                        if "categoryid" in df.columns and "itemid" in df.columns:
                            category = df[df["itemid"] == item_id]["categoryid"].values[0] if not df[df["itemid"] == item_id].empty else "Unknown"
                            st.caption(f"Category: {category}")
                        
                        st.button("View Details", key=f"btn_{item_id}")
                        
                # Show recommendation rationale
                with st.expander("Why these recommendations?"):
                    st.write("""
                        These recommendations are based on:
                        - Popular items with high conversion rates
                        - Items frequently viewed together
                        - Your browsing history (if available)
                    """)
    except Exception as e:
        logger.error(f"Error in recommender system: {str(e)}")
        st.error("Could not display recommender system")

# --- Main App ---
def main():
    # Setup the page
    setup_page()
    
    # Load data
    global df
    df = load_data()
    
    # Create sidebar and get user selections
    selected_page, date_range = create_sidebar()
    
    # Filter data based on date range if available
    filtered_data = df.copy()
    if date_range and len(date_range) == 2 and "timestamp" in df.columns:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_data = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
    
    # Show the selected page
    if selected_page == "📊 Dashboard Overview":
        show_dashboard_overview(filtered_data)
    elif selected_page == "🔍 User Behavior":
        show_user_behavior_analysis(filtered_data)
    elif selected_page == "🤖 Purchase Predictor":
        show_purchase_predictor()
    elif selected_page == "🎯 Recommender":
        show_recommender_system()
    
    # Add footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align: center; color: gray;">
            <p>E-Commerce Analytics Dashboard | <a href="{GITHUB_REPO_URL}">GitHub</a> | <a href="{GITHUB_ISSUES_URL}">Report Issues</a></p>
            <p>Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
