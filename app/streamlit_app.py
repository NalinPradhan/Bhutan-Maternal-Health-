import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_maternal_health_data, validate_data, get_column_summary
from src.preprocessing import handle_missing_values, create_year_categories
from src.utils import plot_trend_over_years, correlation_heatmap, summary_statistics_report

# Page config
st.set_page_config(
    page_title="Bhutan Maternal Health Analytics",
    # page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Bhutan Maternal Health Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**WHO Health Data Analysis & ML Predictions (2000-2023)**")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Data Overview",
    "Exploratory Analysis",
    "ML Predictions",
    "Trends & Insights"
])

# Load data
@st.cache_data
def load_data():
    try:
        df = load_maternal_health_data()
        df_clean = handle_missing_values(df, strategy='interpolate')
        return df, df_clean
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

df_raw, df = load_data()

if df is None:
    st.error("Failed to load data. Please check that data/raw/cleaned_maternal_health_data.csv exists.")
    st.stop()

# ==================== PAGE 1: DATA OVERVIEW ====================
if page == "Data Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Years", len(df))
    with col2:
        st.metric("Indicators", len(df.columns) - 1)
    with col3:
        st.metric("Year Range", f"{df['Year'].min()}-{df['Year'].max()}")
    with col4:
        st.metric("Completeness", f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Data Quality Report")
    col_summary = get_column_summary(df)
    st.dataframe(col_summary, use_container_width=True)
    
    st.subheader("Summary Statistics")
    st.dataframe(summary_statistics_report(df), use_container_width=True)

# ==================== PAGE 2: EXPLORATORY ANALYSIS ====================
elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Time Trends", "🔗 Correlations", "📊 Distributions"])
    
    with tab1:
        st.subheader("Health Indicator Trends Over Time")
        
        indicators = [col for col in df.columns if col != 'Year']
        selected_indicator = st.selectbox("Select Indicator", indicators)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Year'], df[selected_indicator], marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(selected_indicator, fontsize=12)
        ax.set_title(f'{selected_indicator} Trend (2000-2023)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("2000 Value", f"{df[df['Year']==2000][selected_indicator].values[0]:.2f}")
        with col2:
            st.metric("2023 Value", f"{df[df['Year']==2023][selected_indicator].values[0]:.2f}")
        with col3:
            change = df[df['Year']==2023][selected_indicator].values[0] - df[df['Year']==2000][selected_indicator].values[0]
            st.metric("Total Change", f"{change:.2f}")
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Correlation Matrix - Maternal Health Indicators', fontsize=14, fontweight='bold')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Distribution Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('Year')
        selected_col = st.selectbox("Select Column for Distribution", numeric_cols)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(df[selected_col].dropna(), bins=15, edgecolor='black', alpha=0.7)
        ax1.set_xlabel(selected_col, fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'Distribution of {selected_col}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(df[selected_col].dropna())
        ax2.set_ylabel(selected_col, fontsize=11)
        ax2.set_title(f'Box Plot - {selected_col}', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# ==================== PAGE 3: ML PREDICTIONS ====================
elif page == "ML Predictions":
    st.header("Machine Learning Predictions")
    
    # Check if models exist
    if not os.path.exists('models/risk_classifier.pkl'):
        st.warning("⚠️ Models not found. Please run the `03_modeling.ipynb` notebook first to train models.")
        st.info("Run this command: `jupyter notebook notebooks/03_modeling.ipynb`")
        st.stop()
    
    # Load models
    @st.cache_resource
    def load_models():
        classifier = joblib.load('models/risk_classifier.pkl')
        regressor = joblib.load('models/skilled_birth_regressor.pkl')
        class_features = joblib.load('models/classification_features.pkl')
        reg_features = joblib.load('models/regression_features.pkl')
        return classifier, regressor, class_features, reg_features
    
    classifier, regressor, class_features, reg_features = load_models()
    
    st.subheader("Predict Maternal Health Risk Level")
    
    col1, col2 = st.columns(2)
    
    with col1:
        years_since_2000 = st.slider("Years Since 2000", 0, 30, 23)
        skilled_birth = st.slider("Skilled Birth Attendance (%)", 0.0, 100.0, 95.0)
        anc_coverage = st.slider("ANC Coverage - 4+ Visits (%)", 0.0, 100.0, 85.0)
        adolescent_birth = st.slider("Adolescent Birth Rate (per 1000)", 0.0, 100.0, 5.0)
    
    with col2:
        anaemia_prev = st.slider("Anaemia Prevalence (%)", 0.0, 50.0, 32.0)
        healthcare_access = (skilled_birth + 90) / 2  # Simplified
        maternal_risk = (adolescent_birth * 0.5 + anaemia_prev * 0.5)
        skilled_ma3 = skilled_birth
        anaemia_ma3 = anaemia_prev
    
    # Prepare input
    input_data = pd.DataFrame({
        'Years_Since_2000': [years_since_2000],
        'Births attended by skilled health personnel (%)': [skilled_birth],
        'Antenatal care coverage - at least four visits (%)': [anc_coverage],
        'Adolescent birth rate (per 1000 women)': [adolescent_birth],
        'Prevalence of anaemia in women of reproductive age (aged 15-49) (%)': [anaemia_prev],
        'Healthcare_Access_Index': [healthcare_access],
        'Maternal_Risk_Index': [maternal_risk],
        'Skilled_Birth_MA3': [skilled_ma3],
        'Anaemia_MA3': [anaemia_ma3]
    })
    
    if st.button("Predict Risk Level", type="primary"):
        prediction = classifier.predict(input_data)[0]
        probabilities = classifier.predict_proba(input_data)[0]
        
        st.success(f"**Predicted Risk Level: {prediction}**")
        
        # Show probabilities
        prob_df = pd.DataFrame({
            'Risk Level': classifier.classes_,
            'Probability': probabilities * 100
        })
        
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['green', 'orange', 'red']
        ax.barh(prob_df['Risk Level'], prob_df['Probability'], color=colors)
        ax.set_xlabel('Probability (%)', fontsize=11)
        ax.set_title('Risk Level Probabilities', fontsize=12, fontweight='bold')
        st.pyplot(fig)

# ==================== PAGE 4: TRENDS & INSIGHTS ====================
elif page == "Trends & Insights":
    st.header("Key Trends & Insights")
    
    st.subheader("Key Performance Indicators")
    
    # Calculate KPIs
    latest_year = df[df['Year'] == 2023].iloc[0]
    baseline_year = df[df['Year'] == 2000].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        skilled_2023 = latest_year['Births attended by skilled health personnel (%)']
        skilled_2000 = baseline_year['Births attended by skilled health personnel (%)']
        skilled_change = skilled_2023 - skilled_2000
        st.metric(
            "Skilled Birth Attendance",
            f"{skilled_2023:.1f}%",
            f"+{skilled_change:.1f}% since 2000"
        )
    
    with col2:
        adol_2023 = latest_year['Adolescent birth rate (per 1000 women)']
        adol_2000 = baseline_year['Adolescent birth rate (per 1000 women)']
        adol_change = adol_2023 - adol_2000
        st.metric(
            "Adolescent Birth Rate",
            f"{adol_2023:.1f}",
            f"{adol_change:.1f} since 2000",
            delta_color="inverse"
        )
    
    with col3:
        anaemia_2023 = latest_year['Prevalence of anaemia in women of reproductive age (aged 15-49) (%)']
        anaemia_2000 = baseline_year['Prevalence of anaemia in women of reproductive age (aged 15-49) (%)']
        anaemia_change = anaemia_2023 - anaemia_2000
        st.metric(
            "Anaemia Prevalence",
            f"{anaemia_2023:.1f}%",
            f"{anaemia_change:.1f}% since 2000",
            delta_color="inverse"
        )
    
    st.subheader("Progress Overview")
    
    # Multi-line chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    key_indicators = [
        'Births attended by skilled health personnel (%)',
        'Antenatal care coverage - at least four visits (%)',
        'Proportion of births delivered in a health facility (Facility births) (%)'
    ]
    
    for indicator in key_indicators:
        ax.plot(df['Year'], df[indicator], marker='o', label=indicator, linewidth=2)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Maternal Health Service Coverage Trends', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.subheader("Key Insights")
    
    st.info(f"""
    **Major Achievements (2000-2023):**
    - Skilled birth attendance increased from {skilled_2000:.1f}% to {skilled_2023:.1f}%
    - Adolescent birth rate decreased from {adol_2000:.1f} to {adol_2023:.1f} per 1000 women
    - Anaemia prevalence changed from {anaemia_2000:.1f}% to {anaemia_2023:.1f}%
    
    **Overall Assessment:** Bhutan has made significant progress in maternal health services over the past two decades.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Data Source:** WHO Global Health Observatory  
**Project:** Bhutan Healthcare Analytics  
**Year Range:** 2000-2023
""")