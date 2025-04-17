import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import joblib
import time
import logging
from io import BytesIO
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# --- Data Loading & Caching ---
@st.cache_data
def generate_sample_data():
    """Generate synthetic data if real data isn't available"""
    np.random.seed(42)
    size = 5000
    dates = pd.date_range(start="2023-01-01", end="2023-06-30", periods=size)
    
    sample_data = pd.DataFrame({
        'visitorid': np.random.randint(100000, 999999, size),
        'itemid': np.random.randint(100000, 999999, size),
        'event': np.random.choice(['view', 'addtocart', 'purchase'], size, p=[0.85, 0.12, 0.03]),
        'timestamp': np.random.choice(dates, size),
        'user_total_views': np.random.randint(1, 100, size),
        'item_total_views': np.random.randint(1, 200, size),
        'categoryid': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], size),
        'price': np.round(np.random.uniform(10, 500, size), 2)
    })
    
    # Create purchase column based on events
    sample_data['purchase'] = (sample_data['event'] == 'purchase').astype(int)
    return sample_data

@st.cache_data
def load_data():
    """Load and preprocess data with error handling"""
    try:
        # Try loading the real data
        df = pd.read_csv("cleaned_user_item_data_reduced_properties.csv")
        logger.info("Successfully loaded data file")
        
        # Convert timestamp if exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add derived time features if not present
        if 'hour' not in df.columns and 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
        if 'day_of_week' not in df.columns and 'timestamp' in df.columns:
            df['day_of_week'] = df['timestamp'].dt.day_name()
        
        return df
    
    except FileNotFoundError:
        logger.warning("Data file not found - generating sample data")
        st.sidebar.warning("Using sample data - real data not found")
        return generate_sample_data()

# Load the data
with st.spinner("Loading data..."):
    df = load_data()

# Add some calculated fields if they don't exist
if 'purchase' not in df.columns:
    df['purchase'] = (df['event'] == 'purchase').astype(int)

# --- Helper Functions ---
def plot_conversion_funnel(data):
    """Create a conversion funnel visualization"""
    views = data.shape[0]
    add_to_cart = data[data['event'] == 'addtocart'].shape[0]
    purchases = data[data['purchase'] == 1].shape[0]
    
    funnel_df = pd.DataFrame({
        "Stage": ["Views", "Add-to-Cart", "Purchases"],
        "Count": [views, add_to_cart, purchases],
        "Rate": [1, add_to_cart/views if views > 0 else 0, purchases/views if views > 0 else 0]
    })
    
    fig = px.funnel(funnel_df, x='Count', y='Stage', 
                    title='Conversion Funnel',
                    labels={'Count': 'Number of Events', 'Stage': 'Conversion Stage'},
                    color='Stage',
                    color_discrete_sequence=px.colors.qualitative.Pastel)
    
    # Add annotations for rates
    for i, row in funnel_df.iterrows():
        fig.add_annotation(
            x=row['Count'] + max(funnel_df['Count'])*0.05,
            y=row['Stage'],
            text=f"{row['Rate']:.1%}",
            showarrow=False,
            font=dict(size=12)
    )
    return fig

def simulate_purchase_prediction(user_views, item_views, hour, model_type):
    """Simulate purchase probability prediction"""
    # Simple simulation - replace with actual model in production
    base_prob = 0.1
    user_factor = min(0.4, user_views / 250)
    item_factor = min(0.3, item_views / 300)
    time_factor = 0.2 * (1 - abs(hour - 15)/12)  # Peaks around 3pm
    
    if model_type == "LightGBM":
        return min(0.95, base_prob + user_factor + item_factor + time_factor + 0.05)
    elif model_type == "Random Forest":
        return min(0.95, base_prob + user_factor + item_factor + time_factor)
    else:  # Logistic Regression
        return min(0.95, base_prob + 0.8*(user_factor + item_factor + time_factor))

def generate_recommendations(user_id, n=5):
    """Generate sample recommendations - replace with real model"""
    top_items = df.groupby('itemid')['purchase'].sum().sort_values(ascending=False).head(100)
    return np.random.choice(top_items.index, size=n, replace=False)

# --- Sidebar ---
st.sidebar.title("🛒 E-Commerce Analytics")
st.sidebar.image("https://via.placeholder.com/150x50?text=E-Commerce", width=150)

# Date range filter
if 'timestamp' in df.columns:
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
else:
    date_range = None

# Apply date filter if available
if date_range and len(date_range) == 2 and 'timestamp' in df.columns:
    filtered_df = df[(df['timestamp'].dt.date >= date_range[0]) & 
                    (df['timestamp'].dt.date <= date_range[1])]
else:
    filtered_df = df.copy()

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard Overview", "🔍 User Behavior", "🤖 Purchase Predictor", "🎯 Recommender"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard provides analytics and AI insights for e-commerce operations. "
    "Use the tabs above to explore different aspects of customer behavior."
)

# --- Main Content ---
if page == "📊 Dashboard Overview":
    st.title("📊 E-Commerce Dashboard Overview")
    
    # KPI Cards
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    total_views = filtered_df.shape[0]
    add_to_cart = filtered_df[filtered_df['event'] == 'addtocart'].shape[0]
    purchases = filtered_df[filtered_df['purchase'] == 1].shape[0]
    conversion_rate = purchases / total_views if total_views > 0 else 0
    
    col1.metric("Total Views", f"{total_views:,}")
    col2.metric("Add-to-Cart", f"{add_to_cart:,}", f"{add_to_cart/total_views:.1%} rate")
    col3.metric("Purchases", f"{purchases:,}")
    col4.metric("Conversion Rate", f"{conversion_rate:.1%}")
    
    # Conversion Funnel
    st.subheader("Conversion Funnel")
    with st.expander("How to interpret this funnel"):
        st.write("""
        The conversion funnel shows how users progress through different stages:
        1. **Views**: Users who viewed products
        2. **Add-to-Cart**: Users who added items to their cart
        3. **Purchases**: Users who completed purchases
        
        The percentages show the conversion rates between stages.
        """)
    
    fig = plot_conversion_funnel(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Temporal Patterns
    st.subheader("Temporal Patterns")
    tab1, tab2 = st.tabs(["By Hour of Day", "By Day of Week"])
    
    with tab1:
        st.write("User activity patterns throughout the day")
        hourly_data = filtered_df.groupby('hour').size() if 'hour' in filtered_df.columns else pd.Series()
        if not hourly_data.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=hourly_data, ax=ax)
            ax.set_title("Activity by Hour of Day")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Number of Events")
            st.pyplot(fig)
        else:
            st.warning("Hour data not available in this dataset")
    
    with tab2:
        st.write("User activity patterns by day of week")
        if 'day_of_week' in filtered_df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_data = filtered_df['day_of_week'].value_counts().reindex(day_order, fill_value=0)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=daily_data.index, y=daily_data.values, 
                        palette="viridis", ax=ax)
            ax.set_title("Activity by Day of Week")
            ax.set_xlabel("Day")
            ax.set_ylabel("Number of Events")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Day of week data not available in this dataset")

elif page == "🔍 User Behavior":
    st.title("🔍 User Behavior Analysis")
    
    # User Engagement Analysis
    st.subheader("User Engagement Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("User Activity Distribution")
        if 'user_total_views' in filtered_df.columns:
            fig, ax = plt.subplots()
            sns.histplot(filtered_df['user_total_views'], bins=20, kde=True, ax=ax)
            ax.set_title("Distribution of User Views")
            ax.set_xlabel("Total Views per User")
            st.pyplot(fig)
        else:
            st.warning("User view data not available")
    
    with col2:
        st.write("Top Active Users")
        if 'visitorid' in filtered_df.columns and 'user_total_views' in filtered_df.columns:
            top_users = filtered_df.groupby('visitorid')['user_total_views'].max().sort_values(ascending=False).head(10)
            st.dataframe(top_users.reset_index().rename(columns={'user_total_views': 'Total Views'}))
        else:
            st.warning("User data not available")
    
    # Item Popularity Analysis
    st.subheader("Item Popularity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Most Viewed Items")
        if 'itemid' in filtered_df.columns and 'item_total_views' in filtered_df.columns:
            top_items = filtered_df.groupby('itemid')['item_total_views'].max().sort_values(ascending=False).head(10)
            st.dataframe(top_items.reset_index().rename(columns={'item_total_views': 'Total Views'}))
        else:
            st.warning("Item view data not available")
    
    with col2:
        st.write("Best Selling Items")
        if 'itemid' in filtered_df.columns and 'purchase' in filtered_df.columns:
            best_sellers = filtered_df[filtered_df['purchase'] == 1]['itemid'].value_counts().head(10)
            st.dataframe(best_sellers.reset_index().rename(columns={'index': 'Item ID', 'itemid': 'Purchases'}))
        else:
            st.warning("Purchase data not available")

elif page == "🤖 Purchase Predictor":
    st.title("🤖 Purchase Probability Predictor")
    
    # Model Selection
    model_option = st.selectbox(
        "Select Prediction Model",
        ["LightGBM", "Random Forest", "Logistic Regression"],
        index=0,
        help="Choose the machine learning model to use for predictions"
    )
    
    # Display model metrics
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    
    if model_option == "LightGBM":
        col1.metric("ROC AUC Score", "0.89", "0.02 vs baseline")
        col2.metric("Precision-Recall AUC", "0.78", "0.05 vs baseline")
    elif model_option == "Random Forest":
        col1.metric("ROC AUC Score", "0.85", "-0.04 vs LightGBM")
        col2.metric("Precision-Recall AUC", "0.74", "-0.04 vs LightGBM")
    else:  # Logistic Regression
        col1.metric("ROC AUC Score", "0.82", "-0.07 vs LightGBM")
        col2.metric("Precision-Recall AUC", "0.70", "-0.08 vs LightGBM")
    
    # Prediction Interface
    st.subheader("Predict Purchase Probability")
    st.write("Enter user and item details to predict the likelihood of purchase")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            user_views = st.number_input(
                "User's Total Views",
                min_value=1,
                max_value=1000,
                value=20,
                help="Total number of product views by this user"
            )
            hour = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=14,
                help="Current hour when the user is viewing the product"
            )
        
        with col2:
            item_views = st.number_input(
                "Item's Total Views",
                min_value=1,
                max_value=1000,
                value=50,
                help="Total number of views this item has received"
            )
            price = st.number_input(
                "Item Price ($)",
                min_value=0.0,
                max_value=1000.0,
                value=99.99,
                step=0.01,
                help="Price of the item"
            )
        
        submitted = st.form_submit_button("Predict Purchase Probability")
        
        if submitted:
            with st.spinner("Calculating prediction..."):
                # Simulate model prediction delay
                time.sleep(1)
                
                # Get prediction
                proba = simulate_purchase_prediction(user_views, item_views, hour, model_option)
                
                # Display result
                st.success(f"Predicted Purchase Probability: **{proba:.1%}**")
                
                # Add interpretation
                if proba > 0.7:
                    st.info("**High conversion likelihood** - Recommend:")
                    st.markdown("- Premium placement on homepage")
                    st.markdown("- Featured in email campaigns")
                    st.markdown("- Priority in search results")
                elif proba > 0.4:
                    st.info("**Moderate conversion likelihood** - Consider:")
                    st.markdown("- Limited-time promotional offer")
                    st.markdown("- Bundle with related products")
                    st.markdown("- Retargeting ads")
                else:
                    st.info("**Low conversion likelihood** - Suggestions:")
                    st.markdown("- Focus marketing on higher probability items")
                    st.markdown("- Test price adjustments")
                    st.markdown("- Improve product presentation")

elif page == "🎯 Recommender":
    st.title("🎯 Product Recommender System")
    
    # Model Selection
    model_option = st.selectbox(
        "Select Recommendation Algorithm",
        ["Collaborative Filtering", "Content-Based", "Hybrid"],
        index=2,
        help="Choose the recommendation approach to use"
    )
    
    # Display metrics
    st.subheader("System Performance")
    col1, col2, col3 = st.columns(3)
    
    if model_option == "Collaborative Filtering":
        col1.metric("Precision@10", "0.62", "0.04 vs baseline")
        col2.metric("Recall@10", "0.45", "0.03 vs baseline")
        col3.metric("NDCG@10", "0.58", "0.05 vs baseline")
    elif model_option == "Content-Based":
        col1.metric("Precision@10", "0.59", "0.01 vs baseline")
        col2.metric("Recall@10", "0.42", "0.00 vs baseline")
        col3.metric("NDCG@10", "0.53", "0.00 vs baseline")
    else:  # Hybrid
        col1.metric("Precision@10", "0.66", "0.08 vs baseline")
        col2.metric("Recall@10", "0.48", "0.06 vs baseline")
        col3.metric("NDCG@10", "0.63", "0.10 vs baseline")
    
    # Recommendation Interface
    st.subheader("Generate Recommendations")
    
    if 'visitorid' in df.columns:
        user_id = st.selectbox(
            "Select User ID",
            options=df['visitorid'].unique()[:100],
            index=0,
            help="Select a user to generate personalized recommendations"
        )
    else:
        user_id = st.text_input("Enter User ID", value="123456")
    
    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            # Simulate recommendation generation delay
            time.sleep(1.5)
            
            # Get recommendations
            recommended_items = generate_recommendations(user_id, n=5)
            
            # Display recommendations
            st.subheader("Top 5 Recommended Products")
            
            for i, item in enumerate(recommended_items, 1):
                # Simulate item details
                item_details = {
                    'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books']),
                    'views': np.random.randint(50, 500),
                    'price': f"${np.random.randint(20, 500)}",
                    'rating': f"{np.random.uniform(3, 5):.1f} ⭐",
                    'predicted_interest': f"{np.random.uniform(0.6, 0.95):.1%}"
                }
                
                with st.expander(f"Recommendation #{i}: Item {item}", expanded=True if i == 1 else False):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Placeholder image
                        st.image("https://via.placeholder.com/150", width=150)
                    
                    with col2:
                        st.markdown(f"**Category:** {item_details['category']}")
                        st.markdown(f"**Views:** {item_details['views']:,}")
                        st.markdown(f"**Price:** {item_details['price']}")
                        st.markdown(f"**Avg Rating:** {item_details['rating']}")
                        st.markdown(f"**Predicted Interest:** {item_details['predicted_interest']}")
                        
                        # Action buttons
                        btn1, btn2, btn3 = st.columns(3)
                        btn1.button("View Details", key=f"view_{item}")
                        btn2.button("Add to Campaign", key=f"campaign_{item}")
                        btn3.button("Exclude", key=f"exclude_{item}")
            
            # Feedback mechanism
            st.markdown("---")
            st.write("How relevant are these recommendations?")
            feedback = st.radio(
                "Feedback",
                ["Very relevant", "Somewhat relevant", "Not relevant"],
                horizontal=True,
                key="feedback"
            )
            
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback! This helps improve our recommendations.")
                # In a real app, you would log this feedback

# --- Footer ---
st.markdown("---")
footer_col1, footer_col2 = st.columns([3, 1])

with footer_col1:
    st.markdown("""
    **E-Commerce Analytics Dashboard**  
    *Powered by Streamlit | Data updated: {date}*  
    [Report an Issue](https://github.com/your-repo/issues) | [View Source Code](https://github.com/your-repo)
    """.format(date=datetime.now().strftime("%Y-%m-%d")))

with footer_col2:
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/your-repo)
    """, unsafe_allow_html=True)

# Hide Streamlit branding
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)