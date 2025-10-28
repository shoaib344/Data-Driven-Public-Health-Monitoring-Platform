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
    page_title="Hospital Capacity - Public Health Monitor",
    page_icon="🏥",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    return DataSources(), Visualizations(), AlertSystem()

data_sources, viz, alert_system = init_components()

st.title("🏥 Hospital Capacity Dashboard")
st.markdown("Monitor healthcare system capacity, resource availability, and patient flow")

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
    value=(datetime.now() - timedelta(days=14), datetime.now()),
    max_value=datetime.now()
)

# Get hospital capacity data
capacity_data = data_sources.get_hospital_capacity_data(selected_location, date_range)

if not capacity_data.empty:
    # Current capacity status
    st.subheader("🏥 Current Hospital System Status")
    
    latest_data = capacity_data.tail(1).iloc[0]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        bed_occupancy = latest_data['bed_occupancy']
        bed_color = "🟢" if bed_occupancy < 70 else "🟡" if bed_occupancy < 85 else "🔴"
        st.metric(
            label=f"{bed_color} Bed Occupancy",
            value=f"{bed_occupancy}%",
            delta=None
        )
    
    with col2:
        icu_occupancy = latest_data['icu_occupancy']
        icu_color = "🟢" if icu_occupancy < 70 else "🟡" if icu_occupancy < 85 else "🔴"
        st.metric(
            label=f"{icu_color} ICU Occupancy",
            value=f"{icu_occupancy}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Available Beds",
            value=f"{latest_data['available_beds']:,}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Emergency Visits",
            value=f"{latest_data['emergency_visits']:,}",
            delta=None
        )
    
    with col5:
        staff_availability = latest_data['staff_availability']
        staff_color = "🟢" if staff_availability > 80 else "🟡" if staff_availability > 60 else "🔴"
        st.metric(
            label=f"{staff_color} Staff Availability",
            value=f"{staff_availability}%",
            delta=None
        )
    
    # Capacity status interpretation
    st.subheader("📊 System Status Assessment")
    
    # Calculate overall system stress
    stress_factors = []
    stress_score = 0
    
    if bed_occupancy >= 95:
        stress_factors.append("Critical bed occupancy")
        stress_score += 3
    elif bed_occupancy >= 85:
        stress_factors.append("High bed occupancy")
        stress_score += 2
    elif bed_occupancy >= 70:
        stress_factors.append("Elevated bed occupancy")
        stress_score += 1
    
    if icu_occupancy >= 90:
        stress_factors.append("Critical ICU capacity")
        stress_score += 3
    elif icu_occupancy >= 80:
        stress_factors.append("High ICU utilization")
        stress_score += 2
    
    if staff_availability < 60:
        stress_factors.append("Staff shortage")
        stress_score += 2
    elif staff_availability < 80:
        stress_factors.append("Reduced staff availability")
        stress_score += 1
    
    if latest_data['emergency_visits'] > 300:
        stress_factors.append("High emergency department volume")
        stress_score += 1
    
    # Display system status
    if stress_score >= 6:
        st.error("🚨 **CRITICAL SYSTEM STRESS** - Hospital system is at maximum capacity")
        status_color = "red"
    elif stress_score >= 4:
        st.warning("⚠️ **HIGH SYSTEM STRESS** - Hospital system is strained")
        status_color = "orange"
    elif stress_score >= 2:
        st.info("📊 **MODERATE SYSTEM STRESS** - Hospital system is managing well")
        status_color = "yellow"
    else:
        st.success("✅ **NORMAL OPERATIONS** - Hospital system is functioning normally")
        status_color = "green"
    
    if stress_factors:
        st.write("**Contributing factors:**")
        for factor in stress_factors:
            st.write(f"- {factor}")
    
    # Hospital capacity alerts
    st.subheader("🚨 Hospital Capacity Alerts")
    capacity_alerts = alert_system.check_hospital_capacity_alerts(capacity_data, selected_location)
    
    if capacity_alerts:
        for alert in capacity_alerts:
            if alert['severity'] == 'HIGH':
                st.error(f"**{alert['type']}**: {alert['message']}")
            elif alert['severity'] == 'MEDIUM':
                st.warning(f"**{alert['type']}**: {alert['message']}")
            else:
                st.info(f"**{alert['type']}**: {alert['message']}")
    else:
        st.success("No hospital capacity alerts for the selected location and time period.")
    
    # Capacity trends
    st.subheader("📈 Hospital Capacity Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bed = viz.create_time_series(
            capacity_data, 
            x='timestamp', 
            y='bed_occupancy', 
            title='Hospital Bed Occupancy Over Time (%)',
            color='capacity'
        )
        
        # Add threshold lines
        fig_bed.add_hline(y=70, line_dash="dash", line_color="yellow", 
                         annotation_text="Normal Capacity (70%)")
        fig_bed.add_hline(y=85, line_dash="dash", line_color="orange",
                         annotation_text="High Capacity (85%)")
        fig_bed.add_hline(y=95, line_dash="dash", line_color="red",
                         annotation_text="Critical Capacity (95%)")
        
        st.plotly_chart(fig_bed, use_container_width=True)
    
    with col2:
        fig_icu = viz.create_time_series(
            capacity_data, 
            x='timestamp', 
            y='icu_occupancy', 
            title='ICU Occupancy Over Time (%)'
        )
        
        # Add threshold lines for ICU
        fig_icu.add_hline(y=70, line_dash="dash", line_color="yellow", 
                         annotation_text="Normal ICU (70%)")
        fig_icu.add_hline(y=85, line_dash="dash", line_color="orange",
                         annotation_text="High ICU (85%)")
        fig_icu.add_hline(y=95, line_dash="dash", line_color="red",
                         annotation_text="Critical ICU (95%)")
        
        st.plotly_chart(fig_icu, use_container_width=True)
    
    # Multi-metric hospital dashboard
    st.subheader("🏥 Comprehensive Hospital Metrics")
    
    hospital_metrics = ['bed_occupancy', 'icu_occupancy', 'emergency_visits', 'staff_availability']
    available_metrics = [m for m in hospital_metrics if m in capacity_data.columns]
    
    if available_metrics:
        fig_multi = viz.create_multi_metric_chart(
            capacity_data, 
            available_metrics, 
            'Hospital System Metrics Over Time'
        )
        st.plotly_chart(fig_multi, use_container_width=True)
    
    # Capacity utilization analysis
    st.subheader("📊 Capacity Utilization Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Average utilization over time period
        avg_bed_util = capacity_data['bed_occupancy'].mean()
        max_bed_util = capacity_data['bed_occupancy'].max()
        min_bed_util = capacity_data['bed_occupancy'].min()
        
        st.info(f"""
        **Bed Utilization Statistics**
        - Average: {avg_bed_util:.1f}%
        - Peak: {max_bed_util:.1f}%
        - Minimum: {min_bed_util:.1f}%
        - Range: {max_bed_util - min_bed_util:.1f}%
        """)
    
    with col2:
        # ICU statistics
        avg_icu_util = capacity_data['icu_occupancy'].mean()
        max_icu_util = capacity_data['icu_occupancy'].max()
        min_icu_util = capacity_data['icu_occupancy'].min()
        
        st.info(f"""
        **ICU Utilization Statistics**
        - Average: {avg_icu_util:.1f}%
        - Peak: {max_icu_util:.1f}%
        - Minimum: {min_icu_util:.1f}%
        - Range: {max_icu_util - min_icu_util:.1f}%
        """)
    
    with col3:
        # Emergency department statistics
        avg_ed_visits = capacity_data['emergency_visits'].mean()
        max_ed_visits = capacity_data['emergency_visits'].max()
        min_ed_visits = capacity_data['emergency_visits'].min()
        
        st.info(f"""
        **Emergency Dept Statistics**
        - Average: {avg_ed_visits:.0f} visits/day
        - Peak: {max_ed_visits:.0f} visits/day
        - Minimum: {min_ed_visits:.0f} visits/day
        - Range: {max_ed_visits - min_ed_visits:.0f} visits/day
        """)
    
    # Capacity forecasting
    st.subheader("🔮 Short-term Capacity Forecast")
    
    # Simple trend-based forecast
    if len(capacity_data) >= 7:
        # Calculate trend for bed occupancy
        recent_bed_trend = capacity_data.tail(7)['bed_occupancy']
        bed_slope = np.polyfit(range(len(recent_bed_trend)), recent_bed_trend, 1)[0]
        
        # Project next few days
        forecast_days = 3
        current_bed_util = latest_data['bed_occupancy']
        
        forecast_text = f"Based on recent trends:\n"
        
        for day in range(1, forecast_days + 1):
            projected_util = current_bed_util + (bed_slope * day)
            projected_util = max(0, min(100, projected_util))  # Keep within bounds
            
            if projected_util >= 95:
                status = "🔴 Critical"
            elif projected_util >= 85:
                status = "🟡 High"
            else:
                status = "🟢 Normal"
            
            forecast_text += f"- Day +{day}: {projected_util:.1f}% {status}\n"
        
        if abs(bed_slope) < 0.5:
            forecast_text += "\n📊 Capacity levels expected to remain stable"
        elif bed_slope > 0:
            forecast_text += f"\n📈 Capacity trending upward (+{bed_slope:.1f}% per day)"
        else:
            forecast_text += f"\n📉 Capacity trending downward ({bed_slope:.1f}% per day)"
        
        st.info(forecast_text)
    
    # Resource allocation insights
    st.subheader("💡 Resource Management Insights")
    
    insights = []
    
    # Bed management
    if bed_occupancy > 90:
        insights.append("🛏️ **Bed Management**: Consider surge capacity protocols and discharge planning")
    elif bed_occupancy > 80:
        insights.append("🛏️ **Bed Management**: Monitor closely and prepare overflow options")
    
    # ICU management
    if icu_occupancy > 85:
        insights.append("🏥 **ICU Management**: Activate critical care surge protocols")
    elif icu_occupancy > 75:
        insights.append("🏥 **ICU Management**: Review ICU admissions and step-down opportunities")
    
    # Staffing
    if staff_availability < 70:
        insights.append("👩‍⚕️ **Staffing**: Implement contingency staffing plans")
    elif staff_availability < 85:
        insights.append("👩‍⚕️ **Staffing**: Monitor staff wellness and consider support measures")
    
    # Emergency department
    if latest_data['emergency_visits'] > 250:
        insights.append("🚑 **Emergency Dept**: Consider fast-track protocols for non-urgent cases")
    
    if insights:
        for insight in insights:
            st.warning(insight)
    else:
        st.success("✅ **All systems operating within normal parameters**")
    
    # Gauge charts for current status
    st.subheader("⚡ Current Status Gauges")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_bed_gauge = viz.create_gauge_chart(
            bed_occupancy, 
            "Bed Occupancy", 
            max_value=100, 
            thresholds=[70, 85, 95]
        )
        st.plotly_chart(fig_bed_gauge, use_container_width=True)
    
    with col2:
        fig_icu_gauge = viz.create_gauge_chart(
            icu_occupancy, 
            "ICU Occupancy", 
            max_value=100, 
            thresholds=[70, 85, 95]
        )
        st.plotly_chart(fig_icu_gauge, use_container_width=True)
    
    with col3:
        fig_staff_gauge = viz.create_gauge_chart(
            staff_availability, 
            "Staff Availability", 
            max_value=100, 
            thresholds=[60, 80, 100]
        )
        st.plotly_chart(fig_staff_gauge, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Raw Hospital Capacity Data"):
        display_data = capacity_data.tail(30).copy()
        display_data['day_of_week'] = pd.to_datetime(display_data['timestamp']).dt.day_name()
        
        st.dataframe(display_data, use_container_width=True)

else:
    st.warning("No hospital capacity data available for the selected location and date range.")
    
    # Show sample information
    st.info("""
    **Available Data:**
    - Locations: Major metropolitan hospital systems
    - Time Range: Last 30 days of capacity data
    - Metrics: Bed occupancy, ICU capacity, emergency visits, staff availability
    - Updates: Daily capacity reporting from hospital systems
    """)

# Footer information
st.markdown("---")
st.markdown("""
**Data Sources & Methodology:**
- Hospital capacity data from healthcare systems
- Real-time bed management systems integration
- Staff availability from workforce management systems
- Emergency department metrics from patient flow systems
- Threshold alerts based on hospital operations standards
""")
