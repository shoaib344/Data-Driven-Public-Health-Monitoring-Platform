import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from database_data_sources import DatabaseDataSources
from database_alert_system import DatabaseAlertSystem
from visualizations import Visualizations
from auth import (
    check_authentication, show_login_page, show_user_profile, 
    show_user_preferences, show_sidebar_user_info, get_current_user_id,
    has_role, UserRole
)
from api_endpoints import show_data_management_panel, show_data_sources_info
from personalized_dashboard import personalized_dashboard
from ml_interface import ml_interface
from streaming_interface import streaming_interface
from pipeline_interface import pipeline_interface

# Configure page
st.set_page_config(
    page_title="Public Health Monitoring Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize data sources
@st.cache_resource
def init_data_sources():
    return DatabaseDataSources()

@st.cache_resource
def init_alert_system():
    return DatabaseAlertSystem()

@st.cache_resource
def init_visualizations():
    return Visualizations()

def main():
    # Check authentication
    if not check_authentication():
        show_login_page()
        return
    
    # Handle profile/preferences display
    if st.session_state.get('show_profile', False):
        st.title("🏥 Public Health Monitor - Profile")
        
        tab1, tab2 = st.tabs(["Profile", "Preferences"])
        
        with tab1:
            show_user_profile()
        
        with tab2:
            show_user_preferences()
        
        if st.button("← Back to Dashboard"):
            st.session_state.show_profile = False
            st.rerun()
        
        return
    
    # Handle data management display (admin only)
    if st.session_state.get('show_data_management', False):
        if not has_role(UserRole.ADMIN):
            st.error("Access denied. Admin privileges required.")
            return
        
        st.title("🔧 Data Management")
        
        tab1, tab2 = st.tabs(["Data Controls", "Data Sources"])
        
        with tab1:
            show_data_management_panel()
        
        with tab2:
            show_data_sources_info()
        
        if st.button("← Back to Dashboard"):
            st.session_state.show_data_management = False
            st.rerun()
        
        return
    
    # Handle ML dashboard display (health authority and admin only)
    if st.session_state.get('show_ml_dashboard', False):
        ml_interface.show_ml_dashboard()
        
        if st.button("← Back to Dashboard"):
            st.session_state.show_ml_dashboard = False
            st.rerun()
        
        return
    
    # Handle streaming dashboard display (admin only)
    if st.session_state.get('show_streaming_dashboard', False):
        streaming_interface.show_streaming_dashboard()
        
        if st.button("← Back to Dashboard"):
            st.session_state.show_streaming_dashboard = False
            st.rerun()
        
        return
    
    # Handle pipeline dashboard display (admin only)
    if st.session_state.get('show_pipeline_dashboard', False):
        pipeline_interface.show_pipeline_dashboard()
        
        if st.button("← Back to Dashboard"):
            st.session_state.show_pipeline_dashboard = False
            st.rerun()
        
        return
    
    data_sources = init_data_sources()
    alert_system = init_alert_system()
    viz = init_visualizations()
    
    # Main header
    st.title("🏥 Public Health Monitoring Platform")
    st.markdown("Real-time environmental and disease tracking with predictive analytics")
    
    # Show user info in sidebar
    show_sidebar_user_info()
    
    # Add data management for admins
    if has_role(UserRole.ADMIN):
        if st.sidebar.button("🔧 Data Management"):
            st.session_state.show_data_management = True
    
    # Add ML dashboard for health authorities and admins
    if has_role(UserRole.HEALTH_AUTHORITY):
        if st.sidebar.button("🤖 ML Dashboard"):
            st.session_state.show_ml_dashboard = True
    
    # Add streaming dashboard for admins
    if has_role(UserRole.ADMIN):
        if st.sidebar.button("📡 Streaming Dashboard"):
            st.session_state.show_streaming_dashboard = True
    
    # Add pipeline dashboard for admins
    if has_role(UserRole.ADMIN):
        if st.sidebar.button("⚙️ Data Pipeline"):
            st.session_state.show_pipeline_dashboard = True
    
    # Sidebar for location filtering
    st.sidebar.title("📍 Location Filter")
    
    # Get available locations
    locations = data_sources.get_available_locations()
    selected_location = st.sidebar.selectbox(
        "Select Location:",
        options=["All Locations"] + locations,
        index=0
    )
    
    # Date range selector
    st.sidebar.title("📅 Time Range")
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )
    
    # Show personalized dashboard overview
    personalized_dashboard.show_personalized_overview(data_sources, alert_system)
    
    # Show role-specific content
    personalized_dashboard.show_role_specific_content(data_sources, alert_system)
    
    # Show ML predictions for all users
    ml_interface.show_public_predictions(data_sources)
    
    # Show streaming status for health authorities and admins
    if has_role(UserRole.HEALTH_AUTHORITY):
        streaming_interface.show_streaming_status_widget()
    
    # Show pipeline status for admins
    if has_role(UserRole.ADMIN):
        pipeline_interface.show_pipeline_status_widget()
    
    st.markdown("---")
    st.subheader("📊 System Overview")
    
    # Main dashboard overview
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current data for metrics
    current_data = data_sources.get_current_metrics(selected_location)
    
    with col1:
        st.metric(
            label="Air Quality Index",
            value=current_data['aqi'],
            delta=current_data['aqi_change']
        )
    
    with col2:
        st.metric(
            label="Active Disease Cases",
            value=current_data['disease_cases'],
            delta=current_data['disease_change']
        )
    
    with col3:
        st.metric(
            label="Hospital Capacity",
            value=f"{current_data['hospital_capacity']}%",
            delta=f"{current_data['capacity_change']}%"
        )
    
    with col4:
        st.metric(
            label="Environmental Risk",
            value=current_data['risk_level'],
            delta=None
        )
    
    # Alert notifications (personalized for authenticated user)
    st.markdown("---")
    st.subheader("🚨 Current Alerts")
    
    # Get personalized alerts for the current user
    user_id = get_current_user_id()
    alerts = alert_system.get_current_alerts(selected_location)
    
    # TODO: Filter alerts based on user preferences from database
    if alerts:
        for alert in alerts:
            if alert['severity'] == 'HIGH':
                st.error(f"**{alert['type']}**: {alert['message']}")
            elif alert['severity'] == 'MEDIUM':
                st.warning(f"**{alert['type']}**: {alert['message']}")
            else:
                st.info(f"**{alert['type']}**: {alert['message']}")
    else:
        st.success("No active alerts for selected location")
    
    # Quick overview charts
    st.markdown("---")
    st.subheader("📊 Quick Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Air quality trend
        aqi_data = data_sources.get_air_quality_data(selected_location, date_range)
        if not aqi_data.empty:
            fig_aqi = viz.create_time_series(
                aqi_data, 
                x='timestamp', 
                y='aqi', 
                title='Air Quality Index Trend',
                color='aqi'
            )
            st.plotly_chart(fig_aqi, use_container_width=True)
        else:
            st.info("No air quality data available for selected filters")
    
    with col2:
        # Disease cases trend
        disease_data = data_sources.get_disease_data(selected_location, date_range)
        if not disease_data.empty:
            fig_disease = viz.create_time_series(
                disease_data, 
                x='timestamp', 
                y='cases', 
                title='Disease Cases Trend',
                color='cases'
            )
            st.plotly_chart(fig_disease, use_container_width=True)
        else:
            st.info("No disease data available for selected filters")
    
    # Navigation info
    st.markdown("---")
    st.info(
        "📱 **Navigate to specific pages using the sidebar:**\n"
        "- Environmental Data: Detailed air quality and pollution metrics\n"
        "- Disease Monitoring: Outbreak tracking and epidemiological data\n"
        "- Hospital Capacity: Healthcare resource availability\n"
        "- Predictive Analytics: ML-powered forecasting\n"
        "- Geospatial View: Interactive maps and location-based insights\n"
        "- Alert Dashboard: Comprehensive alert management"
    )

if __name__ == "__main__":
    main()
