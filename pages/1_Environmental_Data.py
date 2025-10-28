import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sources import DataSources
from visualizations import Visualizations
from alert_system import AlertSystem

st.set_page_config(
    page_title="Environmental Data - Public Health Monitor",
    page_icon="🌍",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    return DataSources(), Visualizations(), AlertSystem()

data_sources, viz, alert_system = init_components()

st.title("🌍 Environmental Data Dashboard")
st.markdown("Monitor air quality, pollution levels, and environmental health indicators")

# Sidebar filters
st.sidebar.title("Filters")
locations = data_sources.get_available_locations()
selected_location = st.sidebar.selectbox(
    "Select Location:",
    options=["All Locations"] + locations,
    index=0
)

date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(datetime.now() - timedelta(days=7), datetime.now()),
    max_value=datetime.now()
)

# Get environmental data
air_quality_data = data_sources.get_air_quality_data(selected_location, date_range)

if not air_quality_data.empty:
    # Current conditions
    st.subheader("🌡️ Current Air Quality Conditions")
    
    latest_data = air_quality_data.tail(1).iloc[0]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        aqi_value = latest_data['aqi']
        aqi_color = "🟢" if aqi_value <= 50 else "🟡" if aqi_value <= 100 else "🟠" if aqi_value <= 150 else "🔴"
        st.metric(
            label=f"{aqi_color} Air Quality Index",
            value=aqi_value,
            delta=None
        )
    
    with col2:
        st.metric(
            label="PM2.5 (μg/m³)",
            value=latest_data['pm25'],
            delta=None
        )
    
    with col3:
        st.metric(
            label="PM10 (μg/m³)",
            value=latest_data['pm10'],
            delta=None
        )
    
    with col4:
        st.metric(
            label="Ozone (ppb)",
            value=latest_data['ozone'],
            delta=None
        )
    
    with col5:
        st.metric(
            label="NO₂ (ppb)",
            value=latest_data['no2'],
            delta=None
        )
    
    # Air quality interpretation
    st.subheader("📊 Air Quality Health Index")
    aqi = latest_data['aqi']
    
    if aqi <= 50:
        st.success("**Good (0-50)**: Air quality is satisfactory, and air pollution poses little or no risk.")
    elif aqi <= 100:
        st.warning("**Moderate (51-100)**: Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.")
    elif aqi <= 150:
        st.error("**Unhealthy for Sensitive Groups (101-150)**: Members of sensitive groups may experience health effects. The general public is less likely to be affected.")
    elif aqi <= 200:
        st.error("**Unhealthy (151-200)**: Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.")
    else:
        st.error("**Very Unhealthy (201+)**: Health alert: The risk of health effects is increased for everyone.")
    
    # Environmental alerts
    st.subheader("🚨 Environmental Alerts")
    env_alerts = alert_system.check_air_quality_alerts(air_quality_data, selected_location)
    
    if env_alerts:
        for alert in env_alerts:
            if alert['severity'] == 'HIGH':
                st.error(f"**{alert['type']}**: {alert['message']}")
            elif alert['severity'] == 'MEDIUM':
                st.warning(f"**{alert['type']}**: {alert['message']}")
            else:
                st.info(f"**{alert['type']}**: {alert['message']}")
    else:
        st.success("No environmental alerts for the selected location and time period.")
    
    # Time series visualization
    st.subheader("📈 Air Quality Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_aqi = viz.create_time_series(
            air_quality_data, 
            x='timestamp', 
            y='aqi', 
            title='Air Quality Index Over Time',
            color='aqi'
        )
        st.plotly_chart(fig_aqi, use_container_width=True)
    
    with col2:
        fig_pm25 = viz.create_time_series(
            air_quality_data, 
            x='timestamp', 
            y='pm25', 
            title='PM2.5 Levels Over Time'
        )
        st.plotly_chart(fig_pm25, use_container_width=True)
    
    # Multi-pollutant visualization
    st.subheader("🏭 Multiple Pollutant Analysis")
    
    pollutant_metrics = ['pm25', 'pm10', 'ozone', 'no2', 'so2']
    available_metrics = [m for m in pollutant_metrics if m in air_quality_data.columns]
    
    if available_metrics:
        fig_multi = viz.create_multi_metric_chart(
            air_quality_data, 
            available_metrics, 
            'Multiple Pollutant Trends'
        )
        st.plotly_chart(fig_multi, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Pollutant Correlation Analysis")
    
    numeric_columns = air_quality_data.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 1:
        fig_corr = viz.create_correlation_heatmap(
            air_quality_data[numeric_columns], 
            'Pollutant Correlation Matrix'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Distribution analysis
    st.subheader("📊 AQI Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dist = viz.create_distribution_plot(
            air_quality_data, 
            'aqi', 
            'AQI Distribution'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # AQI categories breakdown
        aqi_categories = pd.cut(air_quality_data['aqi'], 
                              bins=[0, 50, 100, 150, 200, 500],
                              labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy'])
        
        category_counts = pd.Series(aqi_categories).value_counts()
        
        if not category_counts.empty:
            import plotly.express as px
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='AQI Category Distribution',
                color_discrete_map={
                    'Good': 'green',
                    'Moderate': 'yellow',
                    'Unhealthy for Sensitive': 'orange',
                    'Unhealthy': 'red',
                    'Very Unhealthy': 'purple'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Raw Environmental Data"):
        st.dataframe(air_quality_data.tail(50), use_container_width=True)
    
    # Health recommendations
    st.subheader("💡 Health Recommendations")
    
    current_aqi = latest_data['aqi']
    
    if current_aqi <= 50:
        st.info("""
        **Recommended Activities:**
        - ✅ All outdoor activities are safe
        - ✅ Perfect time for exercise and recreation
        - ✅ Windows can be opened for fresh air
        """)
    elif current_aqi <= 100:
        st.warning("""
        **Recommended Precautions:**
        - ⚠️ Sensitive individuals should consider limiting prolonged outdoor exertion
        - ✅ Most people can continue normal outdoor activities
        - ⚠️ Consider closing windows during peak pollution hours
        """)
    elif current_aqi <= 150:
        st.error("""
        **Health Precautions:**
        - ❌ Sensitive groups should avoid prolonged outdoor exertion
        - ⚠️ Everyone should limit prolonged outdoor activities
        - ❌ Keep windows closed and use air purifiers if available
        """)
    else:
        st.error("""
        **Health Alert - Take Action:**
        - ❌ Everyone should avoid prolonged outdoor exertion
        - ❌ Stay indoors with windows closed
        - ❌ Use air purifiers and avoid outdoor exercise
        - 🏥 Sensitive individuals should consider staying indoors
        """)

else:
    st.warning("No environmental data available for the selected location and date range.")
    
    # Show sample locations and dates
    st.info("""
    **Available Data:**
    - Locations: New York NY, Los Angeles CA, Chicago IL, Houston TX, Phoenix AZ, and more
    - Time Range: Last 30 days
    - Metrics: AQI, PM2.5, PM10, Ozone, NO₂, SO₂, CO
    """)

# Footer information
st.markdown("---")
st.markdown("""
**Data Sources & Methodology:**
- Air quality data simulated based on EPA monitoring standards
- AQI calculated using standard EPA methodology
- Health recommendations based on official EPA guidelines
- Real-time alerts trigger when thresholds are exceeded
""")
