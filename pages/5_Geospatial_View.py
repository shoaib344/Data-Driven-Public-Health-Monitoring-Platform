import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
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
    page_title="Geospatial View - Public Health Monitor",
    page_icon="🗺️",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    return DataSources(), Visualizations(), AlertSystem()

data_sources, viz, alert_system = init_components()

st.title("🗺️ Geospatial Health Monitoring")
st.markdown("Interactive maps showing health indicators and risk levels across locations")

# Sidebar controls
st.sidebar.title("Map Controls")

map_type = st.sidebar.selectbox(
    "Map Layer:",
    options=["Risk Overview", "Air Quality", "Disease Cases", "Hospital Capacity"],
    index=0
)

show_heatmap = st.sidebar.checkbox("Show Heatmap Overlay", value=True)
show_alerts = st.sidebar.checkbox("Show Alert Locations", value=True)

# Get geospatial data
geospatial_data = data_sources.get_geospatial_data()

if not geospatial_data.empty:
    # Overview metrics
    st.subheader("🌍 Geographic Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk_locations = len(geospatial_data[geospatial_data['risk_level'] == 'HIGH'])
        st.metric(
            label="🔴 High Risk Locations",
            value=high_risk_locations,
            delta=None
        )
    
    with col2:
        medium_risk_locations = len(geospatial_data[geospatial_data['risk_level'] == 'MEDIUM'])
        st.metric(
            label="🟡 Medium Risk Locations",
            value=medium_risk_locations,
            delta=None
        )
    
    with col3:
        avg_aqi = geospatial_data['aqi'].mean()
        st.metric(
            label="🌫️ Average AQI",
            value=f"{avg_aqi:.0f}",
            delta=None
        )
    
    with col4:
        total_cases = geospatial_data['disease_cases'].sum()
        st.metric(
            label="🦠 Total Disease Cases",
            value=f"{total_cases:,}",
            delta=None
        )
    
    # Main map section
    st.subheader(f"🗺️ Interactive Map - {map_type}")
    
    # Create the folium map
    m = viz.create_folium_map(geospatial_data)
    
    # Customize map based on selected layer
    if map_type == "Air Quality":
        # Add AQI-specific styling
        st.info("🌫️ **Air Quality Index**: Green (Good) → Yellow (Moderate) → Orange (Unhealthy for Sensitive) → Red (Unhealthy) → Purple (Very Unhealthy)")
        
    elif map_type == "Disease Cases":
        # Add disease-specific info
        st.info("🦠 **Disease Cases**: Marker size represents relative case volume, colors indicate risk level")
        
    elif map_type == "Hospital Capacity":
        # Add capacity-specific info
        st.info("🏥 **Hospital Capacity**: Higher occupancy percentages shown in warmer colors")
    
    # Display the map
    folium_static(m, width=1200, height=600)
    
    # Location comparison section
    st.subheader("📊 Location Comparison")
    
    # Comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # AQI comparison
        fig_aqi_comparison = px.bar(
            geospatial_data.sort_values('aqi', ascending=True),
            x='aqi',
            y='location',
            orientation='h',
            title='Air Quality Index by Location',
            color='aqi',
            color_continuous_scale='RdYlGn_r'
        )
        fig_aqi_comparison.update_layout(height=500)
        st.plotly_chart(fig_aqi_comparison, use_container_width=True)
    
    with col2:
        # Disease cases comparison
        fig_cases_comparison = px.bar(
            geospatial_data.sort_values('disease_cases', ascending=True),
            x='disease_cases',
            y='location',
            orientation='h',
            title='Disease Cases by Location',
            color='disease_cases',
            color_continuous_scale='Reds'
        )
        fig_cases_comparison.update_layout(height=500)
        st.plotly_chart(fig_cases_comparison, use_container_width=True)
    
    # Risk level distribution
    st.subheader("⚠️ Risk Level Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level pie chart
        risk_counts = geospatial_data['risk_level'].value_counts()
        
        fig_risk_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Risk Level Distribution',
            color_discrete_map={
                'LOW': 'green',
                'MEDIUM': 'orange',
                'HIGH': 'red'
            }
        )
        st.plotly_chart(fig_risk_pie, use_container_width=True)
    
    with col2:
        # Hospital capacity scatter plot
        fig_capacity_scatter = px.scatter(
            geospatial_data,
            x='hospital_capacity',
            y='disease_cases',
            size='population',
            color='risk_level',
            hover_name='location',
            title='Hospital Capacity vs Disease Cases',
            color_discrete_map={
                'LOW': 'green',
                'MEDIUM': 'orange',
                'HIGH': 'red'
            }
        )
        fig_capacity_scatter.update_layout(height=400)
        st.plotly_chart(fig_capacity_scatter, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Geographic Correlation Analysis")
    
    # Calculate correlations between health metrics
    numeric_cols = ['aqi', 'disease_cases', 'hospital_capacity', 'population']
    correlation_matrix = geospatial_data[numeric_cols].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Health Metrics Correlation Matrix",
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Regional analysis
    st.subheader("🏙️ Regional Health Insights")
    
    # Group locations by risk level for analysis
    high_risk_locations = geospatial_data[geospatial_data['risk_level'] == 'HIGH']
    medium_risk_locations = geospatial_data[geospatial_data['risk_level'] == 'MEDIUM']
    low_risk_locations = geospatial_data[geospatial_data['risk_level'] == 'LOW']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error("🔴 **High Risk Locations**")
        if len(high_risk_locations) > 0:
            for _, location in high_risk_locations.iterrows():
                st.write(f"📍 **{location['location']}**")
                st.write(f"   • AQI: {location['aqi']}")
                st.write(f"   • Cases: {location['disease_cases']:,}")
                st.write(f"   • Capacity: {location['hospital_capacity']}%")
                st.write("---")
        else:
            st.write("No high-risk locations")
    
    with col2:
        st.warning("🟡 **Medium Risk Locations**")
        if len(medium_risk_locations) > 0:
            for _, location in medium_risk_locations.iterrows():
                st.write(f"📍 **{location['location']}**")
                st.write(f"   • AQI: {location['aqi']}")
                st.write(f"   • Cases: {location['disease_cases']:,}")
                st.write(f"   • Capacity: {location['hospital_capacity']}%")
                st.write("---")
        else:
            st.write("No medium-risk locations")
    
    with col3:
        st.success("🟢 **Low Risk Locations**")
        if len(low_risk_locations) > 0:
            for _, location in low_risk_locations.iterrows():
                st.write(f"📍 **{location['location']}**")
                st.write(f"   • AQI: {location['aqi']}")
                st.write(f"   • Cases: {location['disease_cases']:,}")
                st.write(f"   • Capacity: {location['hospital_capacity']}%")
                st.write("---")
        else:
            st.write("No low-risk locations")
    
    # Alert locations overlay
    if show_alerts:
        st.subheader("🚨 Alert Locations")
        
        alert_locations = []
        for _, location_data in geospatial_data.iterrows():
            location_alerts = alert_system.get_current_alerts(location_data['location'])
            if location_alerts:
                alert_locations.append({
                    'location': location_data['location'],
                    'alert_count': len(location_alerts),
                    'highest_severity': max([alert['severity'] for alert in location_alerts]),
                    'latitude': location_data['latitude'],
                    'longitude': location_data['longitude']
                })
        
        if alert_locations:
            alert_df = pd.DataFrame(alert_locations)
            
            fig_alert_map = px.scatter_mapbox(
                alert_df,
                lat='latitude',
                lon='longitude',
                size='alert_count',
                color='highest_severity',
                hover_name='location',
                hover_data=['alert_count'],
                color_discrete_map={
                    'HIGH': 'red',
                    'MEDIUM': 'orange',
                    'LOW': 'yellow'
                },
                zoom=3,
                height=400,
                title="Current Alert Locations"
            )
            
            fig_alert_map.update_layout(mapbox_style="open-street-map")
            fig_alert_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
            
            st.plotly_chart(fig_alert_map, use_container_width=True)
        else:
            st.info("No current alerts to display on map")
    
    # Detailed location data table
    with st.expander("📋 Detailed Location Data"):
        # Format the data for better display
        display_data = geospatial_data.copy()
        display_data['AQI'] = display_data['aqi']
        display_data['Disease Cases'] = display_data['disease_cases'].apply(lambda x: f"{x:,}")
        display_data['Hospital Capacity'] = display_data['hospital_capacity'].apply(lambda x: f"{x}%")
        display_data['Population'] = display_data['population'].apply(lambda x: f"{x:,}")
        display_data['Risk Level'] = display_data['risk_level']
        display_data['Location'] = display_data['location']
        
        st.dataframe(
            display_data[['Location', 'AQI', 'Disease Cases', 'Hospital Capacity', 'Population', 'Risk Level']],
            use_container_width=True
        )
    
    # Download data option
    st.subheader("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_data = geospatial_data.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"health_geospatial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON download for GIS applications
        json_data = geospatial_data.to_json(orient='records')
        st.download_button(
            label="🗺️ Download as GeoJSON",
            data=json_data,
            file_name=f"health_geospatial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

else:
    st.error("No geospatial data available")
    st.info("""
    **Geospatial features include:**
    - Interactive maps with health indicators
    - Risk level visualizations by location
    - Heatmap overlays for different metrics
    - Location comparison charts
    - Alert location mapping
    - Correlation analysis between geographic and health factors
    """)

# Legend and information
st.markdown("---")
st.subheader("📖 Map Legend & Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Risk Level Colors:**
    - 🟢 **Low Risk**: Normal conditions, routine monitoring
    - 🟡 **Medium Risk**: Elevated indicators, increased vigilance
    - 🔴 **High Risk**: Critical conditions, immediate attention needed
    
    **Marker Sizes:**
    - Larger markers indicate higher population or case volumes
    - Circle size proportional to relative impact
    """)

with col2:
    st.markdown("""
    **Data Sources:**
    - Air quality monitoring stations
    - Public health surveillance systems
    - Hospital capacity reporting systems
    - Population census data
    
    **Update Frequency:**
    - Real-time for critical alerts
    - Hourly for air quality data
    - Daily for disease and capacity data
    """)

# Footer information
st.markdown("---")
st.markdown("""
**Geospatial Analysis Methodology:**
- Coordinate-based health indicator mapping
- Multi-layer visualization for comprehensive analysis
- Real-time alert integration with geographic context
- Population-weighted risk assessments
- Cross-location correlation analysis for pattern detection
""")
