import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import os
import warnings
import time
import joblib
from pathlib import Path
import re

# Safe SHAP import — fully correct version
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False


# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration with more professional styling
st.set_page_config(
    page_title="🏠 Real Estate House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px;
    }
    .plotly-chart {
        width: 100%;
        height: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #F3F4F6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        border-bottom: 2px solid #2563EB;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar styling
st.sidebar.markdown("""
<div style="text-align: center; padding: 12px 0;">
    <h2 style="color: #1E3A8A;">🏠 Real Estate Analytics</h2>
    <p style="font-size: 0.9rem; color: #6B7280;">Advanced Real Estate Analysis & Predictions</p>
    <hr style="margin: 10px 0;">
</div>
""", unsafe_allow_html=True)

# Theme toggle with better styling
# theme = st.sidebar.selectbox("Interface Theme", ["Light", "Dark", "Modern Blue"])
# if theme == "Dark":
#     st.markdown("""
#     <style>
#     body { background-color: #111827; color: #F9FAFB; }
#     .stApp { background-color: #111827; }
#     .card { background-color: #1F2937; }
#     .metric-card { background-color: #111827; border-left: 4px solid #3B82F6; }
#     .main-header { color: #60A5FA; }
#     .sub-header { color: #93C5FD; }
#     .stTabs [data-baseweb="tab"] { background-color: #374151; }
#     .stTabs [aria-selected="true"] { background-color: #1F2937; border-bottom: 2px solid #3B82F6; }
#     </style>
#     """, unsafe_allow_html=True)
# elif theme == "Modern Blue":
#     st.markdown("""
#     <style>
#     body { background-color: #F0F9FF; color: #0F172A; }
#     .stApp { background-color: #F0F9FF; }
#     .card { background-color: #FFFFFF; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); }
#     .metric-card { background-color: #DBEAFE; border-left: 4px solid #2563EB; }
#     .main-header { color: #1E40AF; }
#     .sub-header { color: #1D4ED8; }
#     .stTabs [data-baseweb="tab"] { background-color: #EFF6FF; }
#     .stTabs [aria-selected="true"] { background-color: #BFDBFE; border-bottom: 2px solid #1D4ED8; }
#     </style>
#     """, unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏠 Real Estate House Price & Analytics Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <p>This comprehensive platform leverages cutting-edge machine learning algorithms for precise real estate price predictions, 
    market analysis, and advanced geospatial visualizations. Drawing from robust historical data spanning 2001-2022, 
    it offers detailed statistical insights, property valuations, and interactive map analytics.</p>
    <p><strong>Ideal for:</strong> Real estate professionals, investors, market analysts, and property developers seeking data-driven decision support.</p>
</div>
""", unsafe_allow_html=True)

# Create cache directory if it doesn't exist
cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)

# Load the dataset with improved caching and preprocessing
@st.cache_data(ttl=3600)
def load_data():
    try:
        start_time = time.time()
        #st.info("Loading and optimizing dataset...")
        
        # Check if processed data cache exists
        processed_file = cache_dir / "processed_real_estate_data.pkl"
        if processed_file.exists():
            data = pd.read_pickle(processed_file)
            st.success(f"Loaded cached data in {time.time() - start_time:.2f} seconds")
            return data
        
        # Load the raw data
        data = pd.read_csv('Real_Estate_Sales_2001-2022_GL.csv', nrows=100000)
        
        # Clean up column names - strip whitespace
        data.columns = [col.strip() for col in data.columns]
        
        # Clean monetary fields
        monetary_cols = ['Assessed Value', 'Sale Amount']
        for col in monetary_cols:
            if col in data.columns:
                data[col] = data[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Clean date fields
        date_cols = ['Date Recorded', 'List Year']
        for col in date_cols:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
        
        # Extract date components
        if 'Date Recorded' in data.columns:
            data['Year'] = data['Date Recorded'].dt.year
            data['Month'] = data['Date Recorded'].dt.month
            data['MonthName'] = data['Date Recorded'].dt.month_name()
            data['Quarter'] = data['Date Recorded'].dt.quarter
            data['Season'] = data['Month'].apply(lambda x: 
                'Winter' if x in [12, 1, 2] else 
                'Spring' if x in [3, 4, 5] else 
                'Summer' if x in [6, 7, 8] else 'Fall')
            data['DayOfWeek'] = data['Date Recorded'].dt.day_name()
        
        # Calculate Sales Ratio if not present
        if 'Sales Ratio' not in data.columns and 'Assessed Value' in data.columns and 'Sale Amount' in data.columns:
            mask = (data['Assessed Value'] > 0) & (data['Sale Amount'] > 0)
            data.loc[mask, 'Sales Ratio'] = data.loc[mask, 'Assessed Value'] / data.loc[mask, 'Sale Amount']
        elif 'Sales Ratio' in data.columns:
            data['Sales Ratio'] = pd.to_numeric(data['Sales Ratio'], errors='coerce')
            
        # Extract coordinates from Location column
        if 'Location' in data.columns:
            def extract_coords(location):
                if pd.isna(location):
                    return None, None
                match = re.search(r'\(([^,]+),\s*([^\)]+)\)', str(location))
                if match:
                    try:
                        return float(match.group(1)), float(match.group(2))
                    except ValueError:
                        return None, None
                return None, None
            
            coords = data['Location'].apply(lambda x: pd.Series(extract_coords(x), index=['Latitude', 'Longitude']))
            data['Latitude'] = coords['Latitude']
            data['Longitude'] = coords['Longitude']
            
        # Create advanced features
        if 'Sale Amount' in data.columns and 'Year' in data.columns:
            yearly_median = data.groupby('Year')['Sale Amount'].median().reset_index()
            yearly_median = yearly_median.rename(columns={'Sale Amount': 'YearlyMedianPrice'})
            data = pd.merge(data, yearly_median, on='Year', how='left')
            data['PriceToYearlyMedian'] = data['Sale Amount'] / data['YearlyMedianPrice']
            
            if 'Town' in data.columns:
                town_median = data.groupby('Town')['Sale Amount'].median().reset_index()
                town_median = town_median.rename(columns={'Sale Amount': 'TownMedianPrice'})
                data = pd.merge(data, town_median, on='Town', how='left')
                data['PriceToTownMedian'] = data['Sale Amount'] / data['TownMedianPrice']
                overall_median = data['Sale Amount'].median()
                data['TownPricePremium'] = data['TownMedianPrice'] / overall_median - 1
        
        if 'Property Type' in data.columns:
            property_median = data.groupby('Property Type')['Sale Amount'].median().reset_index()
            property_median = property_median.rename(columns={'Sale Amount': 'PropertyTypeMedianPrice'})
            data = pd.merge(data, property_median, on='Property Type', how='left')
            data['PriceToPropertyTypeMedian'] = data['Sale Amount'] / data['PropertyTypeMedianPrice']
            
        if 'Sale Amount' in data.columns and all(col in data.columns for col in ['Address', 'Property Type', 'Town']):
            data['PropertyID'] = data['Town'] + '_' + data['Address'] + '_' + data['Property Type']
            data['RepeatSale'] = data.duplicated('PropertyID', keep=False)
            if 'Year' in data.columns:
                data = data.sort_values(['PropertyID', 'Date Recorded'])
                data['DaysSinceLastSale'] = data.groupby('PropertyID')['Date Recorded'].diff().dt.days
                data['PotentialFlip'] = (data['DaysSinceLastSale'] <= 730) & (data['DaysSinceLastSale'] > 0)
                data['PreviousSaleAmount'] = data.groupby('PropertyID')['Sale Amount'].shift(1)
                data['PriceChange'] = data['Sale Amount'] - data['PreviousSaleAmount']
                data['PriceChangePercent'] = (data['PriceChange'] / data['PreviousSaleAmount'] * 100)
                
        critical_cols = ['Sale Amount', 'Property Type', 'Town']
        data = data.dropna(subset=critical_cols)
        
        if 'Sale Amount' in data.columns:
            Q1 = data['Sale Amount'].quantile(0.01)
            Q3 = data['Sale Amount'].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data['Sale Amount'] >= max(lower_bound, 1000)) & 
                        (data['Sale Amount'] <= upper_bound)]
        
        data.to_pickle(processed_file)
        st.success(f"Data processed successfully in {time.time() - start_time:.2f} seconds")
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(columns=['Town', 'Property Type', 'Sale Amount', 'Assessed Value', 'Year'])

with st.spinner('Loading and optimizing data for advanced analytics...'):
    data = load_data()

if data.empty:
    st.error("Failed to load data. Please check your dataset file and try again.")
    st.stop()

#st.sidebar.info(f"Dataset: {data.shape[0]:,} records, {data.shape[1]} features")

st.sidebar.markdown('<div class="sub-header">Search Filters</div>', unsafe_allow_html=True)

with st.sidebar.expander("Location Filters", expanded=True):
    towns = sorted(data['Town'].unique())
    selected_towns = st.multiselect("Towns", towns, default=[])
    if 'Property Type' in data.columns:
        property_types = sorted(data['Property Type'].unique())
        selected_property_types = st.multiselect("Property Types", property_types, default=[])
    else:
        selected_property_types = []
    if 'Residential Type' in data.columns:
        residential_types = sorted(data['Residential Type'].dropna().unique())
        selected_residential_type = st.multiselect("Residential Types", residential_types, default=[])
    else:
        selected_residential_type = []

with st.sidebar.expander("Price & Time Filters", expanded=True):
    min_price = int(data['Sale Amount'].min())
    max_price = int(data['Sale Amount'].max())
    price_scale = st.radio("Price Scale", ["Linear", "Logarithmic"], horizontal=True)
    if price_scale == "Logarithmic":
        log_min = np.log10(max(min_price, 1))
        log_max = np.log10(max_price)
        log_price_range = st.slider("Price Range (Log)", float(log_min), float(log_max), 
                                   (float(log_min), float(log_max)), step=0.1, format="%.1f")
        price_range = (int(10**log_price_range[0]), int(10**log_price_range[1]))
        st.write(f"Price: ${price_range[0]:,} - ${price_range[1]:,}")
    else:
        price_range = st.slider("Price Range ($)", min_price, max_price, (min_price, max_price))
    years = sorted(data['Year'].dropna().unique())
    year_range = st.slider("Year Range", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
    if 'Month' in data.columns:
        months = sorted(data['Month'].dropna().astype(int).unique())
        month_names = [calendar.month_name[m] for m in months if 1 <= m <= 12]
        selected_months = st.multiselect("Months", month_names, default=[])
    else:
        selected_months = []
    if 'Season' in data.columns:
        seasons = sorted(data['Season'].dropna().unique())
        selected_seasons = st.multiselect("Seasons", seasons, default=[])
    else:
        selected_seasons = []

with st.sidebar.expander("Advanced Filters", expanded=False):
    if 'PotentialFlip' in data.columns:
        include_flips = st.checkbox("Include Property Flips", value=True)
    else:
        include_flips = True
    if 'RepeatSale' in data.columns:
        include_repeat_sales = st.checkbox("Include Repeat Sales", value=True)
    else:
        include_repeat_sales = True
    if 'Sales Ratio' in data.columns:
        min_ratio = float(data['Sales Ratio'].min())
        max_ratio = float(data['Sales Ratio'].max())
        ratio_range = st.slider("Sales Ratio Range", min_ratio, max_ratio, (min_ratio, max_ratio))
    else:
        ratio_range = (0, 100)

if st.sidebar.button("Reset All Filters", key="reset_button", use_container_width=True):
    selected_towns = []
    selected_property_types = []
    selected_residential_type = []
    selected_months = []
    selected_seasons = []
    price_range = (min_price, max_price)
    year_range = (int(min(years)), int(max(years)))
    ratio_range = (float(data['Sales Ratio'].min()), float(data['Sales Ratio'].max())) if 'Sales Ratio' in data.columns else (0, 100)
    include_flips = True
    include_repeat_sales = True

with st.spinner('Applying filters and preparing analysis...'):
    filtered_data = data.copy()
    if selected_towns:
        filtered_data = filtered_data[filtered_data['Town'].isin(selected_towns)]
    if selected_property_types:
        filtered_data = filtered_data[filtered_data['Property Type'].isin(selected_property_types)]
    if selected_residential_type and 'Residential Type' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Residential Type'].isin(selected_residential_type)]
    filtered_data = filtered_data[
        (filtered_data['Sale Amount'] >= price_range[0]) & 
        (filtered_data['Sale Amount'] <= price_range[1])
    ]
    filtered_data = filtered_data[
        (filtered_data['Year'] >= year_range[0]) & 
        (filtered_data['Year'] <= year_range[1])
    ]
    if selected_months and 'MonthName' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['MonthName'].isin(selected_months)]
    if selected_seasons and 'Season' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Season'].isin(selected_seasons)]
    if 'Sales Ratio' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['Sales Ratio'] >= ratio_range[0]) & 
            (filtered_data['Sales Ratio'] <= ratio_range[1])
        ]
    if 'PotentialFlip' in filtered_data.columns and not include_flips:
        filtered_data = filtered_data[~filtered_data['PotentialFlip']]
    if 'RepeatSale' in filtered_data.columns and not include_repeat_sales:
        filtered_data = filtered_data[~filtered_data['RepeatSale']]

st.markdown('<div class="sub-header">Dataset Insights</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Properties", f"{len(filtered_data):,}")
col2.metric("Median Price", f"${filtered_data['Sale Amount'].median():,.0f}")
col3.metric("Towns", f"{filtered_data['Town'].nunique()}")
col4.metric("Property Types", f"{filtered_data['Property Type'].nunique()}")

if len(filtered_data) < 50:
    st.warning("⚠️ Not enough data points for reliable analysis. Please broaden your search criteria.")

def prepare_model_data(data):
    categorical_features = []
    numeric_features = ['Assessed Value']
    date_features = []
    if 'Town' in data.columns:
        categorical_features.append('Town')
    if 'Property Type' in data.columns:
        categorical_features.append('Property Type')
    if 'Residential Type' in data.columns:
        categorical_features.append('Residential Type')
    if 'Year' in data.columns:
        date_features.append('Year')
    if 'Month' in data.columns:
        date_features.append('Month')
    if 'Season' in data.columns:
        categorical_features.append('Season')
    if 'PriceToYearlyMedian' in data.columns:
        numeric_features.append('PriceToYearlyMedian')
    if 'TownPricePremium' in data.columns:
        numeric_features.append('TownPricePremium')
    
    all_features = categorical_features + numeric_features + date_features
    X = data[all_features].copy()
    y = data['Sale Amount']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('date', 'passthrough', date_features)
        ])
    
    return X, y, preprocessor

@st.cache_resource
def train_model(data):
    X, y, preprocessor = prepare_model_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]),
        'Extra Trees': Pipeline([
            ('preprocessor', preprocessor),
            ('model', ExtraTreesRegressor(n_estimators=100, random_state=42))
        ]),
        'Ridge Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=1.0, random_state=42))
        ])
    }
    
    results = {}
    feature_importances = {}
    shap_values = {}
    
    try:
        for name, pipeline in models.items():
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            if name in ['Random Forest', 'Extra Trees']:
                tree_preds = np.array([tree.predict(preprocessor.transform(X_test)) 
                                    for tree in pipeline.named_steps['model'].estimators_])
                lower_bound = np.percentile(tree_preds, 5, axis=0)
                upper_bound = np.percentile(tree_preds, 95, axis=0)
            else:
                residuals = y_test - y_pred
                residual_std = np.std(residuals)
                lower_bound = y_pred - 1.96 * residual_std
                upper_bound = y_pred + 1.96 * residual_std
            
            within_interval = ((y_test >= lower_bound) & (y_test <= upper_bound)).mean() * 100
            training_time = time.time() - start_time
            
            results[name] = {
                'model': pipeline,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'y_test': y_test,
                'y_pred': y_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'prediction_interval_coverage': within_interval,
                'training_time': training_time
            }
            
            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                feature_importances[name] = pipeline.named_steps['model'].feature_importances_
            
            if name == 'Random Forest' and len(X_test) > 0:
                try:
                    X_shap = X_test.iloc[:min(100, len(X_test))]
                    X_shap_processed = preprocessor.transform(X_shap)
                    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
                    shap_values[name] = explainer.shap_values(X_shap_processed)
                except Exception as e:
                    st.warning(f"Could not calculate SHAP values: {e}")
                    shap_values[name] = None
            
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        return results, best_model[0], preprocessor, feature_importances, shap_values
    except Exception as e:
        st.error(f"Error training models: {e}")
        return {}, "None", None, {}, {}

if len(filtered_data) >= 50:
    with st.spinner('Training advanced prediction models...'):
        model_results, best_model_name, preprocessor, feature_importances, shap_values = train_model(filtered_data)
    
    tabs = st.tabs([
        "📊 Market Overview", 
        "🔮 Price Estimations", 
        "📈 Market Trends",
        "🌎 Geographic Analysis", 
        "📉 Model Performance", 
        "💡 Advanced Insights"
    ])
    
    with tabs[0]:
        st.markdown('<div class="sub-header">📊 Market Overview</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Price", f"${filtered_data['Sale Amount'].mean():,.0f}")
            st.metric("Price Volatility", f"{filtered_data['Sale Amount'].std() / filtered_data['Sale Amount'].mean() * 100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Median Price", f"${filtered_data['Sale Amount'].median():,.0f}")
            st.metric("Price Range", f"${filtered_data['Sale Amount'].quantile(0.25):,.0f} - ${filtered_data['Sale Amount'].quantile(0.75):,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'Year' in filtered_data.columns:
                yearly_growth = filtered_data.groupby('Year')['Sale Amount'].median().pct_change() * 100
                avg_yearly_growth = yearly_growth.mean()
                st.metric("Avg. Yearly Growth", f"{avg_yearly_growth:.1f}%")
            if 'RepeatSale' in filtered_data.columns:
                repeat_sales_pct = filtered_data['RepeatSale'].mean() * 100
                st.metric("Repeat Sales", f"{repeat_sales_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">Price Distribution</div>', unsafe_allow_html=True)
        price_dist_chart_type = st.radio(
            "Select Distribution Chart", 
            ["Histogram", "KDE", "Box Plot"],
            horizontal=True
        )
        if price_dist_chart_type == "Histogram":
            fig = px.histogram(
                filtered_data, 
                x="Sale Amount",
                nbins=50,
                color="Property Type" if len(filtered_data['Property Type'].unique()) <= 5 else None,
                marginal="box",
                opacity=0.7,
                title="Price Distribution by Property Type"
            )
            fig.update_layout(
                xaxis_title="Sale Amount ($)",
                yaxis_title="Count",
                legend_title="Property Type",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        elif price_dist_chart_type == "KDE":
            fig, ax = plt.subplots(figsize=(10, 6))
            prop_types = filtered_data['Property Type'].unique()
            if len(prop_types) <= 5:
                for prop_type in prop_types:
                    subset = filtered_data[filtered_data['Property Type'] == prop_type]
                    sns.kdeplot(subset['Sale Amount'], label=prop_type, ax=ax, fill=True, alpha=0.3)
            else:
                sns.kdeplot(filtered_data['Sale Amount'], ax=ax, fill=True)
            ax.set_title("Price Density Distribution")
            ax.set_xlabel("Sale Amount ($)")
            ax.set_ylabel("Density")
            if len(prop_types) <= 5:
                ax.legend(title="Property Type")
            st.pyplot(fig)
        else:
            fig = px.box(
                filtered_data,
                x="Property Type" if len(filtered_data['Property Type'].unique()) <= 8 else "Town",
                y="Sale Amount",
                color="Property Type" if len(filtered_data['Property Type'].unique()) <= 8 else None,
                title="Price Distribution by Property Type",
                points="outliers"
            )
            fig.update_layout(
                xaxis_title="Property Type" if len(filtered_data['Property Type'].unique()) <= 8 else "Town",
                yaxis_title="Sale Amount ($)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="sub-header">Town Price Comparison</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            if len(filtered_data['Town'].unique()) <= 15:
                fig = px.bar(
                    filtered_data.groupby('Town')['Sale Amount'].median().reset_index(),
                    x='Town',
                    y='Sale Amount',
                    color='Town',
                    labels={'Sale Amount': 'Median Sale Price ($)'},
                    title="Median Sale Price by Town"
                )
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Too many towns to display. Please filter your data.")
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Top Towns by Price**")
            top_towns = filtered_data.groupby('Town')['Sale Amount'].median().sort_values(ascending=False).head(5)
            for town, price in top_towns.items():
                st.markdown(f"• {town}: ${price:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Most Active Towns**")
            active_towns = filtered_data['Town'].value_counts().head(5)
            for town, count in active_towns.items():
                st.markdown(f"• {town}: {count:,} sales")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown('<div class="sub-header">🔮 Price Predictions</div>', unsafe_allow_html=True)
        if best_model_name != "None":
            st.markdown(f"""
            <div class="card">
                <p>This price prediction model uses {best_model_name} to estimate property values 
                based on multiple factors including location, property characteristics, and market conditions.</p>
                <p><strong>Model Accuracy:</strong> {model_results[best_model_name]['r2']*100:.1f}% 
                (R² Score)</p>
            </div>
            """, unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Property Value Estimator")
                towns = sorted(filtered_data['Town'].unique())
                property_types = sorted(filtered_data['Property Type'].unique())
                with st.form("property_value_form"):
                    town = st.selectbox("Town", towns)
                    prop_type = st.selectbox("Property Type", property_types)
                    if 'Assessed Value' in filtered_data.columns:
                        avg_assessed = filtered_data[filtered_data['Town'] == town]['Assessed Value'].median()
                        assessed_value = st.number_input("Assessed Value", 
                                                        value=int(avg_assessed), 
                                                        step=1000)
                    else:
                        assessed_value = st.number_input("Assessed Value", 
                                                        value=100000, 
                                                        step=1000)
                    if 'Year' in filtered_data.columns:
                        year = st.selectbox("Year", sorted(filtered_data['Year'].unique()))
                    else:
                        year = 2022
                    if 'Month' in filtered_data.columns:
                        month = st.selectbox("Month", range(1, 13))
                    else:
                        month = 6
                    submit_button = st.form_submit_button("Estimate Value")
                if submit_button and best_model_name != "None":
                    sample = pd.DataFrame({
                        'Town': [town],
                        'Property Type': [prop_type],
                        'Assessed Value': [assessed_value],
                        'Year': [year],
                        'Month': [month]
                    })
                    for col in preprocessor.feature_names_in_:
                        if col not in sample.columns:
                            sample[col] = 0
                    model = model_results[best_model_name]['model']
                    prediction = model.predict(sample)[0]
                    prediction_interval = model_results[best_model_name]['mape'] / 100 * prediction
                    lower_bound = prediction - prediction_interval
                    upper_bound = prediction + prediction_interval
                    st.success(f"Estimated Value: ${prediction:,.0f}")
                    st.info(f"Prediction Range: ${lower_bound:,.0f} - ${upper_bound:,.0f}")
                    comparables = filtered_data[
                        (filtered_data['Town'] == town) & 
                        (filtered_data['Property Type'] == prop_type)
                    ].copy()
                    if len(comparables) > 0:
                        comparables['SimilarityScore'] = abs(comparables['Sale Amount'] - prediction) / prediction
                        similar_props = comparables.sort_values('SimilarityScore').head(5)
                        st.markdown("##### Comparable Properties")
                        for _, prop in similar_props.iterrows():
                            st.markdown(f"• ${prop['Sale Amount']:,.0f} - {prop['Town']}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### What Impacts Property Values?")
                if best_model_name in feature_importances:
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        try:
                            feature_names = preprocessor.get_feature_names_out()
                        except:
                            feature_names = [f"feature_{i}" for i in range(len(feature_importances[best_model_name]))]
                    else:
                        feature_names = [f"feature_{i}" for i in range(len(feature_importances[best_model_name]))]
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importances[best_model_name]
                    })
                    importance_df['Feature'] = importance_df['Feature'].str.replace('cat__', '')
                    importance_df['Feature'] = importance_df['Feature'].str.replace('num__', '')
                    top_features = importance_df.sort_values('Importance', ascending=False).head(10)
                    fig = px.bar(
                        top_features,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top Factors Affecting Property Value"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                if best_model_name in shap_values and shap_values[best_model_name] is not None:
                    st.markdown("#### Property Value Drivers (SHAP Analysis)")
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(
                            shap_values[best_model_name], 
                            feature_names=feature_names,
                            plot_type="bar",
                            show=False
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"Could not create SHAP plot: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown('<div class="sub-header">📈 Market Trends</div>', unsafe_allow_html=True)
        if 'Year' in filtered_data.columns:
            st.markdown("#### Price Trends Over Time")
            trend_data = filtered_data.groupby('Year')['Sale Amount'].agg(['median', 'mean', 'count']).reset_index()
            trend_data = trend_data.rename(columns={'median': 'Median Price', 'mean': 'Average Price', 'count': 'Sales Volume'})
            col1, col2 = st.columns([3, 1])
            with col1:
                trend_metric = st.radio(
                    "Price Metric", 
                    ["Median Price", "Average Price"],
                    horizontal=True
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_data['Year'],
                    y=trend_data[trend_metric],
                    mode='lines+markers',
                    name=trend_metric,
                    line=dict(color='#2563EB', width=3)
                ))
                fig.add_trace(go.Bar(
                    x=trend_data['Year'],
                    y=trend_data['Sales Volume'],
                    name='Sales Volume',
                    marker_color='rgba(37, 99, 235, 0.2)',
                    opacity=0.7,
                    yaxis='y2'
                ))
                fig.update_layout(
                    title=f"{trend_metric} and Sales Volume by Year",
                    xaxis=dict(title='Year'),
                    yaxis=dict(
                        title=dict(
                            text=f"{trend_metric} ($)",
                            font=dict(color='#2563EB')
                        ),
                        tickfont=dict(color='#2563EB')
                    ),
                    yaxis2=dict(
                        title=dict(
                            text='Sales Volume',
                            font=dict(color='#64748B')
                        ),
                        tickfont=dict(color='#64748B'),
                        anchor='x',
                        overlaying='y',
                        side='right'
                    ),
                    height=500,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Market Summary**")
                first_year = trend_data['Year'].min()
                last_year = trend_data['Year'].max()
                first_price = trend_data[trend_data['Year'] == first_year][trend_metric].values[0]
                last_price = trend_data[trend_data['Year'] == last_year][trend_metric].values[0]
                total_growth = (last_price / first_price - 1) * 100
                years_diff = last_year - first_year
                annualized_growth = ((last_price / first_price) ** (1 / max(1, years_diff)) - 1) * 100
                st.metric("Total Growth", f"{total_growth:.1f}%")
                st.metric("Annualized Growth", f"{annualized_growth:.1f}%")
                if len(trend_data) >= 3:
                    recent_data = trend_data.iloc[-3:]
                    recent_trend = (recent_data[trend_metric].iloc[-1] / recent_data[trend_metric].iloc[0] - 1) * 100
                    st.metric("Recent Trend (3 years)", f"{recent_trend:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Highest Growth Years**")
                trend_data['Growth'] = trend_data[trend_metric].pct_change() * 100
                top_growth_years = trend_data.dropna().sort_values('Growth', ascending=False).head(3)
                for _, year_data in top_growth_years.iterrows():
                    st.markdown(f"• {int(year_data['Year'])}: {year_data['Growth']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
        if 'Month' in filtered_data.columns:
            st.markdown("#### Seasonal Trends")
            seasonal_data = filtered_data.groupby('Month')['Sale Amount'].median().reset_index()
            seasonal_data['Month'] = seasonal_data['Month'].astype(int)
            seasonal_data['MonthName'] = seasonal_data['Month'].apply(lambda x: calendar.month_name[x])
            seasonal_data = seasonal_data.sort_values('Month')
            fig = px.line(
                seasonal_data,
                x='MonthName',
                y='Sale Amount',
                markers=True,
                title="Median Sale Price by Month",
                labels={'Sale Amount': 'Median Sale Price ($)', 'MonthName': 'Month'}
            )
            seasonal_data['Change'] = seasonal_data['Sale Amount'].pct_change() * 100
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Peak Season**")
                peak_month_idx = seasonal_data['Sale Amount'].idxmax()
                peak_month = seasonal_data.loc[peak_month_idx, 'MonthName']
                peak_price = seasonal_data.loc[peak_month_idx, 'Sale Amount']
                trough_month_idx = seasonal_data['Sale Amount'].idxmin()
                trough_month = seasonal_data.loc[trough_month_idx, 'MonthName']
                trough_price = seasonal_data.loc[trough_month_idx, 'Sale Amount']
                seasonal_diff = (peak_price / trough_price - 1) * 100
                st.markdown(f"**Peak Month:** {peak_month}")
                st.markdown(f"**Lowest Month:** {trough_month}")
                st.markdown(f"**Seasonal Price Difference:** {seasonal_diff:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Sales Volume by Season**")
                if 'Season' in filtered_data.columns:
                    season_counts = filtered_data['Season'].value_counts()
                    fig = px.pie(
                        values=season_counts.values,
                        names=season_counts.index,
                        title="Sales Distribution by Season"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown('<div class="sub-header">🌎 Geographic Analysis</div>', unsafe_allow_html=True)
        
        # Calculate median price per town for color coding
        town_prices = filtered_data.groupby('Town')['Sale Amount'].median().reset_index()
        town_counts = filtered_data['Town'].value_counts().reset_index()
        town_counts.columns = ['Town', 'Count']
        town_data = pd.merge(town_prices, town_counts, on='Town')
        
        # Define approximate coordinates for towns (you may need to replace this with real data)
        town_coords = {
            town: (41.6 + i * 0.01, -72.7 + i * 0.01)  # Example coords for Connecticut towns
            for i, town in enumerate(town_data['Town'])
        }
        
        if len(town_data) > 0:
            # Create map
            m = folium.Map(
                location=[41.6, -72.7],  # Center of Connecticut as default
                zoom_start=9,
                tiles="OpenStreetMap"
            )
            
            # Normalize prices for color gradient
            min_price = town_data['Sale Amount'].min()
            max_price = town_data['Sale Amount'].max()
            price_range = max_price - min_price
            
            # Add markers for each town with color based on price
            for _, row in town_data.iterrows():
                town = row['Town']
                price = row['Sale Amount']
                if price_range > 0:
                    norm_price = (price - min_price) / price_range
                else:
                    norm_price = 0.5  # Default if all prices are the same
                
                # Define color based on normalized price
                if norm_price >= 0.75:
                    color = 'red'  # High price
                elif norm_price >= 0.5:
                    color = 'orange'  # Medium-high price
                elif norm_price >= 0.25:
                    color = 'yellow'  # Medium-low price
                else:
                    color = 'green'  # Low price
                
                # Get coordinates (use real ones if available)
                lat, lon = town_coords.get(town, (41.6, -72.7))
                
                # Add marker
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    tooltip=f"{town}: ${price:,.0f} ({row['Count']} sales)"
                ).add_to(m)
            
            # Display map
            st.markdown("#### Town Price Map")
            folium_static(m, width=1000, height=600)
            
            # Show town price table
            st.markdown("#### Town Price Statistics")
            town_data_display = town_data.copy()
            town_data_display['Sale Amount'] = town_data_display['Sale Amount'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(town_data_display, hide_index=True)
        else:
            st.warning("No town data available with the current filters.")
    
    with tabs[4]:
        st.markdown('<div class="sub-header">📉 Model Performance</div>', unsafe_allow_html=True)
        if best_model_name != "None":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Model", best_model_name, help=f"Selected model: {best_model_name}")
            with col2:
                st.metric("R² Score", f"{model_results[best_model_name]['r2'] * 100:.2f}%")
            with col3:
                st.metric("RMSE", f"{model_results[best_model_name]['rmse']:,.0f}", help="Root Mean Squared Error")
            with col4:
                st.metric("MAPE", f"{model_results[best_model_name]['mape']:.2f}%", help="Mean Absolute Percentage Error")
            
            st.markdown("#### Model Comparison")
            model_comparison = pd.DataFrame({
                'Model': list(model_results.keys()),
                'R² Score': [result['r2'] * 100 for result in model_results.values()],
                'RMSE': [result['rmse'] for result in model_results.values()],
                'MAE': [result['mae'] for result in model_results.values()],
                'MAPE (%)': [result['mape'] for result in model_results.values()],
                'Interval Coverage (%)': [result['prediction_interval_coverage'] for result in model_results.values()],
                'Training Time (s)': [result['training_time'] for result in model_results.values()]
            })
            model_comparison['R² Score'] = model_comparison['R² Score'].apply(lambda x: f"{x:.2f}%")
            model_comparison['RMSE'] = model_comparison['RMSE'].apply(lambda x: f"{x:,.0f}")
            model_comparison['MAE'] = model_comparison['MAE'].apply(lambda x: f"{x:,.0f}")
            model_comparison['MAPE (%)'] = model_comparison['MAPE (%)'].apply(lambda x: f"{x:.2f}%")
            model_comparison['Interval Coverage (%)'] = model_comparison['Interval Coverage (%)'].apply(lambda x: f"{x:.2f}%")
            model_comparison['Training Time (s)'] = model_comparison['Training Time (s)'].apply(lambda x: f"{x:.2f}")
            st.dataframe(model_comparison, hide_index=True)
            
            st.markdown("#### Prediction Accuracy")
            if len(model_results[best_model_name]['y_test']) > 1000:
                indices = np.random.choice(len(model_results[best_model_name]['y_test']), 1000, replace=False)
                y_test_sample = model_results[best_model_name]['y_test'].iloc[indices]
                y_pred_sample = model_results[best_model_name]['y_pred'][indices]
                lower_bound_sample = model_results[best_model_name]['lower_bound'][indices]
                upper_bound_sample = model_results[best_model_name]['upper_bound'][indices]
            else:
                y_test_sample = model_results[best_model_name]['y_test']
                y_pred_sample = model_results[best_model_name]['y_pred']
                lower_bound_sample = model_results[best_model_name]['lower_bound']
                upper_bound_sample = model_results[best_model_name]['upper_bound']
            
            fig = go.Figure()
            max_val = max(y_test_sample.max(), y_pred_sample.max())
            min_val = min(y_test_sample.min(), y_pred_sample.min())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='gray', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=y_test_sample,
                y=y_pred_sample,
                mode='markers',
                name='Predictions',
                marker=dict(
                    color='#2563EB',
                    size=8,
                    opacity=0.6
                )
            ))
            fig.update_layout(
                title="Predicted vs Actual Sale Prices",
                xaxis=dict(title="Actual Price ($)"),
                yaxis=dict(
                    title=dict(
                        text="Predicted Price ($)",
                        font=dict(color='#2563EB')
                    )
                ),
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Residual Analysis")
            residuals = y_test_sample - y_pred_sample
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    residuals,
                    nbins=50,
                    title="Residual Distribution",
                    labels={"value": "Residual ($)"}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(
                    x=y_pred_sample,
                    y=residuals,
                    title="Residuals vs Predicted Values",
                    labels={"x": "Predicted Price ($)", "y": "Residual ($)"}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Prediction Intervals")
            interval_df = pd.DataFrame({
                'Actual': y_test_sample,
                'Predicted': y_pred_sample,
                'Lower Bound': lower_bound_sample,
                'Upper Bound': upper_bound_sample
            })
            interval_df['Within Interval'] = (
                (interval_df['Actual'] >= interval_df['Lower Bound']) &
                (interval_df['Actual'] <= interval_df['Upper Bound'])
            )
            coverage = interval_df['Within Interval'].mean() * 100
            st.metric("Prediction Interval Coverage", f"{coverage:.2f}%")
            st.markdown("##### Sample Predictions with Intervals")
            sample_display = interval_df.sample(min(10, len(interval_df)))
            for col in ['Actual', 'Predicted', 'Lower Bound', 'Upper Bound']:
                sample_display[col] = sample_display[col].apply(lambda x: f"${x:,.0f}")
            st.dataframe(sample_display, hide_index=True)
        else:
            st.warning("No models have been trained on this dataset yet.")
    
    with tabs[5]:
        st.markdown('<div class="sub-header">💡 Advanced Insights</div>', unsafe_allow_html=True)
        st.markdown("#### Raw Data")
        st.dataframe(filtered_data, hide_index=True, height=400)
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        csv = convert_df(filtered_data)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="real_estate_data.csv",
            mime="text/csv",
        )
        st.markdown("#### Data Summary")
        summary = filtered_data.describe().T
        summary['count'] = summary['count'].astype(int)
        for col in ['mean', '50%', 'min', 'max']:
            if col in summary.columns:
                summary[col] = summary[col].apply(
                    lambda x: f"${x:,.0f}" if (isinstance(x, (int, float)) and x > 1000) else 
                              f"{x:.2f}" if isinstance(x, (int, float)) else 
                              str(x)
                )
        st.dataframe(summary, height=300)
        numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            st.markdown("#### Correlation Matrix")
            corr = filtered_data[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    def main():
        pass
    main()
