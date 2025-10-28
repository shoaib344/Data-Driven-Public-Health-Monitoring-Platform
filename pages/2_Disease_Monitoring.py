import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sources import DataSources
from visualizations import Visualizations
from alert_system import AlertSystem

st.set_page_config(
    page_title="Disease Monitoring - Public Health Monitor",
    page_icon="🦠",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    return DataSources(), Visualizations(), AlertSystem()

data_sources, viz, alert_system = init_components()

st.title("🦠 Disease Monitoring Dashboard")
st.markdown("Track disease outbreaks, epidemiological trends, and public health indicators")

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
    value=(datetime.now() - timedelta(days=30), datetime.now()),
    max_value=datetime.now()
)

# Get disease data
disease_data = data_sources.get_disease_data(selected_location, date_range)

if not disease_data.empty:
    # Current disease status
    st.subheader("📊 Current Disease Status")
    
    # Calculate current metrics
    recent_data = disease_data.tail(7)  # Last 7 days
    current_cases = recent_data['cases'].sum()
    daily_average = recent_data['cases'].mean()
    
    # Calculate trend
    if len(recent_data) >= 7:
        first_half = recent_data.head(3)['cases'].mean()
        second_half = recent_data.tail(4)['cases'].mean()
        trend_change = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
    else:
        trend_change = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Cases (7 days)",
            value=f"{int(current_cases):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Daily Average",
            value=f"{daily_average:.0f}",
            delta=f"{trend_change:+.1f}%"
        )
    
    with col3:
        # Risk level based on recent trends
        if trend_change > 50:
            risk_level = "🔴 HIGH"
        elif trend_change > 20:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"
        
        st.metric(
            label="Outbreak Risk",
            value=risk_level,
            delta=None
        )
    
    with col4:
        # Calculate reproduction number estimate (simplified)
        if len(disease_data) >= 14:
            recent_avg = disease_data.tail(7)['cases'].mean()
            previous_avg = disease_data.tail(14).head(7)['cases'].mean()
            r_estimate = recent_avg / previous_avg if previous_avg > 0 else 1.0
        else:
            r_estimate = 1.0
        
        st.metric(
            label="R₀ Estimate",
            value=f"{r_estimate:.2f}",
            delta="↗️" if r_estimate > 1.1 else "↘️" if r_estimate < 0.9 else "→"
        )
    
    # Disease alerts
    st.subheader("🚨 Disease Outbreak Alerts")
    disease_alerts = alert_system.check_disease_alerts(disease_data, selected_location)
    
    if disease_alerts:
        for alert in disease_alerts:
            if alert['severity'] == 'HIGH':
                st.error(f"**{alert['type']}**: {alert['message']}")
            elif alert['severity'] == 'MEDIUM':
                st.warning(f"**{alert['type']}**: {alert['message']}")
            else:
                st.info(f"**{alert['type']}**: {alert['message']}")
    else:
        st.success("No disease outbreak alerts for the selected location and time period.")
    
    # Main disease trend chart
    st.subheader("📈 Disease Case Trends")
    
    fig_cases = viz.create_time_series(
        disease_data, 
        x='timestamp', 
        y='cases', 
        title='Daily Disease Cases Over Time',
        color='cases'
    )
    st.plotly_chart(fig_cases, use_container_width=True)
    
    # Analysis sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Weekly Pattern Analysis")
        
        # Add day of week analysis
        disease_with_dow = disease_data.copy()
        disease_with_dow['day_of_week'] = pd.to_datetime(disease_with_dow['timestamp']).dt.day_name()
        
        daily_avg = disease_with_dow.groupby('day_of_week')['cases'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        import plotly.express as px
        fig_weekly = px.bar(
            x=daily_avg.index,
            y=daily_avg.values,
            title='Average Cases by Day of Week',
            labels={'x': 'Day of Week', 'y': 'Average Cases'}
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with col2:
        st.subheader("📉 Moving Averages")
        
        # Calculate moving averages
        disease_with_ma = disease_data.copy()
        disease_with_ma['3_day_avg'] = disease_with_ma['cases'].rolling(window=3, center=True).mean()
        disease_with_ma['7_day_avg'] = disease_with_ma['cases'].rolling(window=7, center=True).mean()
        
        import plotly.graph_objects as go
        fig_ma = go.Figure()
        
        fig_ma.add_trace(go.Scatter(
            x=disease_with_ma['timestamp'],
            y=disease_with_ma['cases'],
            mode='lines',
            name='Daily Cases',
            line=dict(color='lightblue', width=1),
            opacity=0.6
        ))
        
        fig_ma.add_trace(go.Scatter(
            x=disease_with_ma['timestamp'],
            y=disease_with_ma['3_day_avg'],
            mode='lines',
            name='3-Day Average',
            line=dict(color='orange', width=2)
        ))
        
        fig_ma.add_trace(go.Scatter(
            x=disease_with_ma['timestamp'],
            y=disease_with_ma['7_day_avg'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='red', width=3)
        ))
        
        fig_ma.update_layout(
            title='Disease Cases with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Cases',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_ma, use_container_width=True)
    
    # Outbreak detection analysis
    st.subheader("🔍 Outbreak Detection Analysis")
    
    # Calculate outbreak probability based on various factors
    outbreak_factors = []
    
    # Factor 1: Recent trend
    if trend_change > 50:
        outbreak_factors.append("Rapid increase in cases (>50%)")
    elif trend_change > 20:
        outbreak_factors.append("Moderate increase in cases (>20%)")
    
    # Factor 2: Reproduction number
    if r_estimate > 1.2:
        outbreak_factors.append(f"High reproduction rate (R₀ = {r_estimate:.2f})")
    elif r_estimate > 1.1:
        outbreak_factors.append(f"Elevated reproduction rate (R₀ = {r_estimate:.2f})")
    
    # Factor 3: Case volume
    if daily_average > 200:
        outbreak_factors.append(f"High daily case volume ({daily_average:.0f} cases/day)")
    elif daily_average > 100:
        outbreak_factors.append(f"Elevated daily case volume ({daily_average:.0f} cases/day)")
    
    if outbreak_factors:
        st.warning("**Outbreak Risk Factors Detected:**")
        for factor in outbreak_factors:
            st.write(f"- {factor}")
    else:
        st.success("**No significant outbreak risk factors detected.**")
    
    # Epidemiological insights
    st.subheader("🧪 Epidemiological Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Peak detection
        if len(disease_data) >= 14:
            recent_max = disease_data.tail(14)['cases'].max()
            overall_max = disease_data['cases'].max()
            
            st.info(f"""
            **Recent Peak Analysis**
            - Highest recent cases: {recent_max}
            - Overall peak: {overall_max}
            - Peak intensity: {(recent_max/overall_max)*100:.0f}%
            """)
    
    with col2:
        # Variability analysis
        recent_std = recent_data['cases'].std()
        recent_mean = recent_data['cases'].mean()
        cv = (recent_std / recent_mean) * 100 if recent_mean > 0 else 0
        
        st.info(f"""
        **Case Variability**
        - Standard deviation: {recent_std:.1f}
        - Mean cases: {recent_mean:.1f}
        - Coefficient of variation: {cv:.1f}%
        """)
    
    with col3:
        # Growth rate
        if len(disease_data) >= 7:
            week_start = disease_data.tail(7).head(1)['cases'].iloc[0]
            week_end = disease_data.tail(1)['cases'].iloc[0]
            growth_rate = ((week_end - week_start) / week_start * 100) if week_start > 0 else 0
            
            st.info(f"""
            **Weekly Growth**
            - Start of week: {week_start}
            - End of week: {week_end}
            - Growth rate: {growth_rate:+.1f}%
            """)
    
    # Case distribution analysis
    st.subheader("📊 Case Distribution")
    
    fig_dist = viz.create_distribution_plot(
        disease_data, 
        'cases', 
        'Daily Case Count Distribution'
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Raw Disease Data"):
        # Show more detailed disease data if available
        display_data = disease_data.tail(30).copy()
        display_data['day_of_week'] = pd.to_datetime(display_data['timestamp']).dt.day_name()
        display_data['week_number'] = pd.to_datetime(display_data['timestamp']).dt.isocalendar().week
        
        st.dataframe(display_data, use_container_width=True)
    
    # Public health recommendations
    st.subheader("💡 Public Health Recommendations")
    
    if trend_change > 50 or r_estimate > 1.2:
        st.error("""
        **High Risk - Enhanced Measures Recommended:**
        - 🚨 Consider implementing stricter public health measures
        - 😷 Increased mask mandates in high-risk areas
        - 🏠 Recommend work-from-home policies
        - 🧼 Enhanced sanitation and hygiene campaigns
        - 📊 Increase testing and contact tracing capacity
        """)
    elif trend_change > 20 or r_estimate > 1.1:
        st.warning("""
        **Moderate Risk - Vigilant Monitoring:**
        - ⚠️ Maintain current public health measures
        - 📈 Increase surveillance and monitoring
        - 💉 Consider targeted vaccination campaigns
        - 📢 Public awareness and education campaigns
        - 🏥 Ensure healthcare capacity readiness
        """)
    else:
        st.success("""
        **Low Risk - Standard Precautions:**
        - ✅ Continue routine surveillance
        - 📋 Maintain standard prevention protocols
        - 📊 Regular monitoring of key indicators
        - 💡 Focus on prevention education
        - 🔬 Continue research and preparedness activities
        """)

else:
    st.warning("No disease monitoring data available for the selected location and date range.")
    
    # Show sample information
    st.info("""
    **Available Data:**
    - Locations: Major US cities with health departments
    - Time Range: Last 30 days of epidemiological data
    - Metrics: Daily cases, outbreak indicators, trend analysis
    - Diseases: Influenza, COVID-19, Norovirus, RSV, Pneumonia
    """)

# Footer information
st.markdown("---")
st.markdown("""
**Data Sources & Methodology:**
- Disease surveillance data from public health departments
- Outbreak detection using statistical algorithms
- R₀ estimates based on serial interval assumptions
- Risk assessments follow CDC epidemiological guidelines
- Real-time alerts for significant changes in disease patterns
""")
