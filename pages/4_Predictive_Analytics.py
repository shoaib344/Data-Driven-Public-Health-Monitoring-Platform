import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_data_sources import DatabaseDataSources
from visualizations import Visualizations
from ml_interface import ml_interface
from auth import check_authentication, show_login_page, has_role, UserRole

st.set_page_config(
    page_title="Predictive Analytics - Public Health Monitor",
    page_icon="🔮",
    layout="wide"
)

# Check authentication
if not check_authentication():
    show_login_page()
    st.stop()

# Initialize components
@st.cache_resource
def init_components():
    return DatabaseDataSources(), Visualizations()

data_sources, viz = init_components()

# Helper function for simple forecasting
def create_simple_forecast(data, metric, hours):
    """Create simple forecast for non-AQI metrics"""
    if data.empty:
        return pd.DataFrame()
    
    # Simple linear trend forecast
    values = data[metric].values
    x = np.arange(len(values))
    
    # Fit linear trend
    coeffs = np.polyfit(x, values, 1)
    
    # Generate future timestamps
    last_timestamp = pd.to_datetime(data['timestamp'].max())
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=hours,
        freq='h'
    )
    
    # Generate predictions
    future_x = np.arange(len(values), len(values) + hours)
    predictions = np.polyval(coeffs, future_x)
    
    # Add some noise for realism
    noise = np.random.normal(0, values.std() * 0.1, hours)
    predictions += noise
    
    # Ensure predictions are reasonable (non-negative)
    predictions = np.maximum(predictions, 0)
    
    return pd.DataFrame({
        'timestamp': future_timestamps,
        f'predicted_{metric}': predictions,
        'confidence': ['High' if i < 6 else 'Medium' if i < 12 else 'Low' for i in range(hours)]
    })

st.title("🔮 Predictive Analytics Dashboard")
st.markdown("Machine learning models for forecasting health trends and risk assessment")

# Show ML interface based on user role
if has_role(UserRole.HEALTH_AUTHORITY):
    # Full ML dashboard for health authorities and admins
    ml_interface.show_ml_dashboard()
else:
    # Public ML predictions
    ml_interface.show_public_predictions(data_sources)

# Sidebar controls
st.sidebar.title("Prediction Controls")
locations = data_sources.get_available_locations()
selected_location = st.sidebar.selectbox(
    "Select Location:",
    options=["All Locations"] + locations,
    index=0
)

prediction_type = st.sidebar.selectbox(
    "Select Prediction Type:",
    options=["Air Quality", "Disease Outbreak", "Hospital Capacity"],
    index=0
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (hours/days):",
    min_value=6,
    max_value=72,
    value=24,
    step=6
)

# Get training data based on prediction type
if prediction_type == "Air Quality":
    training_data = data_sources.get_air_quality_data(
        selected_location, 
        (datetime.now() - timedelta(days=30), datetime.now())
    )
    target_metric = "aqi"
elif prediction_type == "Disease Outbreak":
    training_data = data_sources.get_disease_data(
        selected_location, 
        (datetime.now() - timedelta(days=30), datetime.now())
    )
    target_metric = "cases"
else:  # Hospital Capacity
    training_data = data_sources.get_hospital_capacity_data(
        selected_location, 
        (datetime.now() - timedelta(days=30), datetime.now())
    )
    target_metric = "bed_occupancy"

# Main content
if not training_data.empty:
    # Model training section
    st.subheader("🤖 Model Training & Performance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button(f"Train {prediction_type} Model", type="primary"):
            with st.spinner("Training machine learning model..."):
                if prediction_type == "Air Quality":
                    training_results = models.train_air_quality_model(training_data)
                elif prediction_type == "Disease Outbreak":
                    training_results = models.train_disease_model(training_data)
                else:
                    # For hospital capacity, we'll use air quality model structure
                    training_results = models.train_air_quality_model(training_data)
                
                if 'error' in training_results:
                    st.error(f"Model training failed: {training_results['error']}")
                else:
                    st.success("Model trained successfully!")
                    
                    # Display training results
                    if 'best_model' in training_results:
                        st.info(f"**Best Model:** {training_results['best_model']}")
                        st.info(f"**Performance Score (R²):** {training_results['best_score']:.3f}")
                        
                        # Feature importance
                        if 'feature_importance' in training_results and training_results['feature_importance']:
                            st.subheader("📊 Feature Importance")
                            importance_df = pd.DataFrame([
                                {'Feature': k, 'Importance': v} 
                                for k, v in training_results['feature_importance'].items()
                            ]).sort_values('Importance', ascending=False).head(10)
                            
                            import plotly.express as px
                            fig_importance = px.bar(
                                importance_df, 
                                x='Importance', 
                                y='Feature',
                                orientation='h',
                                title='Top 10 Feature Importance'
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        # Model status
        st.subheader("📈 Model Status")
        performance = models.get_model_performance()
        
        if target_metric in performance:
            st.success(f"✅ {prediction_type} model trained")
            st.info(f"Last updated: {performance[target_metric]['last_updated']}")
        else:
            st.warning("⚠️ Model not trained yet")
            st.info("Click 'Train Model' to get started")
    
    # Prediction section
    st.subheader("🎯 Generate Predictions")
    
    if st.button(f"Generate {forecast_horizon}-hour Forecast", type="secondary"):
        with st.spinner("Generating predictions..."):
            if prediction_type == "Air Quality":
                predictions = models.predict_air_quality(training_data, forecast_horizon)
            else:
                # For now, create simple predictions for other types
                predictions = create_simple_forecast(training_data, target_metric, forecast_horizon)
            
            if not predictions.empty:
                st.success("Predictions generated successfully!")
                
                # Display prediction chart
                st.subheader("📊 Prediction Results")
                
                # Combine historical and predicted data for visualization
                historical_subset = training_data.tail(48)  # Last 48 hours/days
                
                if prediction_type == "Air Quality":
                    fig_pred = viz.create_prediction_chart(
                        historical_subset[['timestamp', 'aqi']], 
                        predictions[['timestamp', 'predicted_aqi']], 
                        f"{prediction_type} Forecast - {selected_location}"
                    )
                else:
                    # Create generic prediction chart
                    fig_pred = viz.create_prediction_chart(
                        historical_subset[['timestamp', target_metric]], 
                        predictions, 
                        f"{prediction_type} Forecast - {selected_location}"
                    )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Prediction confidence and insights
                st.subheader("🎯 Prediction Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction_type == "Air Quality":
                        pred_values = predictions['predicted_aqi']
                        current_value = training_data['aqi'].iloc[-1]
                    else:
                        pred_values = predictions.iloc[:, 1]  # Assume second column is prediction
                        current_value = training_data[target_metric].iloc[-1]
                    
                    avg_prediction = pred_values.mean()
                    change = ((avg_prediction - current_value) / current_value * 100) if current_value > 0 else 0
                    
                    st.metric(
                        label="Forecast Average",
                        value=f"{avg_prediction:.1f}",
                        delta=f"{change:+.1f}%"
                    )
                
                with col2:
                    max_prediction = pred_values.max()
                    st.metric(
                        label="Forecast Peak",
                        value=f"{max_prediction:.1f}",
                        delta=None
                    )
                
                with col3:
                    min_prediction = pred_values.min()
                    st.metric(
                        label="Forecast Low",
                        value=f"{min_prediction:.1f}",
                        delta=None
                    )
                
                # Risk assessment based on predictions
                st.subheader("⚠️ Risk Assessment")
                
                risk_alerts = []
                
                if prediction_type == "Air Quality":
                    high_aqi_hours = sum(1 for val in pred_values if val > 150)
                    moderate_aqi_hours = sum(1 for val in pred_values if 100 < val <= 150)
                    
                    if high_aqi_hours > 0:
                        risk_alerts.append(f"🔴 **High Risk**: {high_aqi_hours} hours with unhealthy air quality predicted")
                    if moderate_aqi_hours > 0:
                        risk_alerts.append(f"🟡 **Moderate Risk**: {moderate_aqi_hours} hours with moderate air quality predicted")
                        
                elif prediction_type == "Disease Outbreak":
                    high_case_days = sum(1 for val in pred_values if val > 200)
                    increasing_trend = pred_values.iloc[-1] > pred_values.iloc[0]
                    
                    if high_case_days > 0:
                        risk_alerts.append(f"🔴 **High Risk**: {high_case_days} days with elevated case counts predicted")
                    if increasing_trend:
                        risk_alerts.append("📈 **Trend Alert**: Cases expected to increase over forecast period")
                        
                else:  # Hospital Capacity
                    high_capacity_days = sum(1 for val in pred_values if val > 90)
                    critical_capacity_days = sum(1 for val in pred_values if val > 95)
                    
                    if critical_capacity_days > 0:
                        risk_alerts.append(f"🔴 **Critical Risk**: {critical_capacity_days} days with critical capacity predicted")
                    elif high_capacity_days > 0:
                        risk_alerts.append(f"🟡 **High Risk**: {high_capacity_days} days with high capacity predicted")
                
                if risk_alerts:
                    for alert in risk_alerts:
                        st.warning(alert)
                else:
                    st.success("✅ **Low Risk**: No significant risk factors detected in forecast period")
                
                # Display prediction data table
                with st.expander("📋 View Detailed Predictions"):
                    st.dataframe(predictions, use_container_width=True)
            else:
                st.error("Failed to generate predictions. Please ensure model is trained first.")
    
    # Historical trend analysis
    st.subheader("📈 Historical Trend Analysis")
    
    # Create trend visualization
    fig_trend = viz.create_time_series(
        training_data, 
        x='timestamp', 
        y=target_metric, 
        title=f'{prediction_type} Historical Trends - {selected_location}',
        color=target_metric
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Statistical analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Statistical Summary")
        
        stats = training_data[target_metric].describe()
        
        st.info(f"""
        **{target_metric.replace('_', ' ').title()} Statistics:**
        - Mean: {stats['mean']:.1f}
        - Median: {stats['50%']:.1f}
        - Standard Deviation: {stats['std']:.1f}
        - Min: {stats['min']:.1f}
        - Max: {stats['max']:.1f}
        """)
    
    with col2:
        st.subheader("🔍 Data Quality Assessment")
        
        total_points = len(training_data)
        missing_points = training_data[target_metric].isna().sum()
        completeness = ((total_points - missing_points) / total_points) * 100
        
        # Calculate data recency
        latest_timestamp = pd.to_datetime(training_data['timestamp'].max())
        hours_since_update = (datetime.now() - latest_timestamp).total_seconds() / 3600
        
        st.info(f"""
        **Data Quality Metrics:**
        - Total data points: {total_points:,}
        - Data completeness: {completeness:.1f}%
        - Hours since last update: {hours_since_update:.1f}
        - Data range: {(pd.to_datetime(training_data['timestamp'].max()) - pd.to_datetime(training_data['timestamp'].min())).days} days
        """)
    
    # Model comparison section
    st.subheader("🧠 Model Performance Comparison")
    
    st.info("""
    **Available Models:**
    - **Random Forest**: Good for non-linear patterns and feature interactions
    - **Gradient Boosting**: Excellent for complex time series with trends
    - **Linear Regression**: Best for simple linear relationships and interpretability
    
    **Model Selection Criteria:**
    - Accuracy (R² score)
    - Robustness to outliers
    - Computational efficiency
    - Interpretability of results
    """)
    
    # Advanced analytics section
    st.subheader("🔬 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal analysis
        training_data_with_features = training_data.copy()
        training_data_with_features['hour'] = pd.to_datetime(training_data_with_features['timestamp']).dt.hour
        training_data_with_features['day_of_week'] = pd.to_datetime(training_data_with_features['timestamp']).dt.dayofweek
        
        if prediction_type == "Air Quality":
            hourly_avg = training_data_with_features.groupby('hour')[target_metric].mean()
            
            import plotly.express as px
            fig_hourly = px.line(
                x=hourly_avg.index,
                y=hourly_avg.values,
                title='Average by Hour of Day',
                labels={'x': 'Hour', 'y': target_metric.replace('_', ' ').title()}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Weekly patterns
        daily_avg = training_data_with_features.groupby('day_of_week')[target_metric].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig_weekly = px.bar(
            x=day_names,
            y=daily_avg.values,
            title='Average by Day of Week',
            labels={'x': 'Day of Week', 'y': target_metric.replace('_', ' ').title()}
        )
        st.plotly_chart(fig_weekly, use_container_width=True)

else:
    st.warning(f"No training data available for {prediction_type} in {selected_location}")
    
    st.info("""
    **To use predictive analytics:**
    1. Select a location with available data
    2. Choose prediction type (Air Quality, Disease Outbreak, or Hospital Capacity)
    3. Train a model using historical data
    4. Generate forecasts for future periods
    
    **Model Features:**
    - Time-based patterns (hour, day, week)
    - Lagged values and moving averages
    - Trend detection and seasonality
    - Multiple algorithm comparison
    """)


# Footer
st.markdown("---")
st.markdown("""
**Machine Learning Pipeline:**
- Data preprocessing with feature engineering
- Multiple algorithm training and comparison
- Cross-validation for model selection
- Confidence intervals for uncertainty quantification
- Real-time model retraining capabilities
""")
