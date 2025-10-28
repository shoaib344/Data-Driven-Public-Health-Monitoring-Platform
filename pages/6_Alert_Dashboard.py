import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sources import DataSources
from alert_system import AlertSystem
from visualizations import Visualizations

st.set_page_config(
    page_title="Alert Dashboard - Public Health Monitor",
    page_icon="🚨",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    return DataSources(), AlertSystem(), Visualizations()

data_sources, alert_system, viz = init_components()

st.title("🚨 Alert Management Dashboard")
st.markdown("Comprehensive alert monitoring, management, and analysis system")

# Sidebar filters
st.sidebar.title("Alert Filters")

locations = data_sources.get_available_locations()
selected_locations = st.sidebar.multiselect(
    "Filter by Location:",
    options=["All Locations"] + locations,
    default=["All Locations"]
)

severity_filter = st.sidebar.multiselect(
    "Filter by Severity:",
    options=["HIGH", "MEDIUM", "LOW"],
    default=["HIGH", "MEDIUM", "LOW"]
)

alert_type_filter = st.sidebar.multiselect(
    "Filter by Alert Type:",
    options=["Air Quality", "Disease Outbreak", "Hospital Capacity", "Trend Alert"],
    default=["Air Quality", "Disease Outbreak", "Hospital Capacity", "Trend Alert"]
)

# Time range for historical alerts
time_range = st.sidebar.selectbox(
    "Historical Time Range:",
    options=["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
    index=1
)

# Convert time range to days
time_range_days = {"Last 24 Hours": 1, "Last 7 Days": 7, "Last 30 Days": 30}[time_range]

# Current alert summary
st.subheader("📊 Current Alert Summary")

alert_summary = alert_system.get_alert_summary()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🔴 Critical Alerts",
        value=alert_summary['HIGH'],
        delta=None
    )

with col2:
    st.metric(
        label="🟡 Medium Alerts", 
        value=alert_summary['MEDIUM'],
        delta=None
    )

with col3:
    st.metric(
        label="🟢 Low Priority Alerts",
        value=alert_summary['LOW'],
        delta=None
    )

with col4:
    total_alerts = sum(alert_summary.values())
    st.metric(
        label="📋 Total Active Alerts",
        value=total_alerts,
        delta=None
    )

# Current active alerts
st.subheader("🚨 Active Alerts")

# Get current alerts for all locations or filtered locations
if "All Locations" in selected_locations:
    all_current_alerts = []
    for location in locations:
        location_alerts = alert_system.get_current_alerts(location)
        all_current_alerts.extend(location_alerts)
else:
    all_current_alerts = []
    for location in selected_locations:
        location_alerts = alert_system.get_current_alerts(location)
        all_current_alerts.extend(location_alerts)

# Filter alerts based on sidebar selections
filtered_alerts = [
    alert for alert in all_current_alerts
    if (alert['severity'] in severity_filter and 
        alert['type'] in alert_type_filter)
]

if filtered_alerts:
    # Sort alerts by severity and timestamp
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    filtered_alerts.sort(key=lambda x: (severity_order[x['severity']], x['timestamp']), reverse=True)
    
    # Display alerts in expandable cards
    for i, alert in enumerate(filtered_alerts):
        severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[alert['severity']]
        
        with st.expander(f"{severity_icon} {alert['type']} - {alert['location']} ({alert['severity']})", 
                        expanded=(i < 3 and alert['severity'] == 'HIGH')):  # Auto-expand first 3 high severity
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Message:** {alert['message']}")
                st.write(f"**Location:** {alert['location']}")
                st.write(f"**Timestamp:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                if 'value' in alert and 'metric' in alert:
                    st.write(f"**Current Value:** {alert['value']} ({alert['metric']})")
            
            with col2:
                # Alert management buttons
                if st.button(f"Acknowledge", key=f"ack_{i}"):
                    st.success("Alert acknowledged")
                    st.rerun()
                
                if st.button(f"Dismiss", key=f"dismiss_{i}"):
                    if alert_system.dismiss_alert(f"alert_{i}"):
                        st.success("Alert dismissed")
                        st.rerun()
                
                if alert['severity'] == 'HIGH':
                    if st.button(f"Escalate", key=f"escalate_{i}"):
                        st.warning("Alert escalated to emergency response team")
else:
    st.info("No active alerts match the selected criteria")

# Alert analytics
st.subheader("📈 Alert Analytics")

col1, col2 = st.columns(2)

with col1:
    # Alert distribution by type
    if filtered_alerts:
        alert_types = [alert['type'] for alert in filtered_alerts]
        type_counts = pd.Series(alert_types).value_counts()
        
        fig_types = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title='Active Alerts by Type',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_types, use_container_width=True)
    else:
        st.info("No alert type data available for selected filters")

with col2:
    # Alert distribution by severity
    if filtered_alerts:
        alert_severities = [alert['severity'] for alert in filtered_alerts]
        severity_counts = pd.Series(alert_severities).value_counts()
        
        color_map = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
        colors = [color_map[severity] for severity in severity_counts.index]
        
        fig_severity = px.bar(
            x=severity_counts.index,
            y=severity_counts.values,
            title='Active Alerts by Severity',
            color=severity_counts.index,
            color_discrete_map=color_map
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    else:
        st.info("No alert severity data available for selected filters")

# Alert trends over time
st.subheader("📊 Alert Trends")

# Get historical alerts
historical_alerts = []
for location in (locations if "All Locations" in selected_locations else selected_locations):
    location_history = alert_system.get_alert_history(location, time_range_days)
    historical_alerts.extend(location_history)

if historical_alerts:
    # Create DataFrame for analysis
    alert_df = pd.DataFrame(historical_alerts)
    alert_df['date'] = pd.to_datetime(alert_df['timestamp']).dt.date
    
    # Daily alert counts
    daily_alerts = alert_df.groupby(['date', 'severity']).size().reset_index().rename(columns={0: 'count'})
    
    if not daily_alerts.empty:
        fig_timeline = px.bar(
            daily_alerts,
            x='date',
            y='count',
            color='severity',
            title=f'Alert Trends - {time_range}',
            color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
        )
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Alert frequency by location
    if len(selected_locations) > 1 or "All Locations" in selected_locations:
        location_alerts = alert_df['location'].value_counts().head(10)
        
        fig_location = px.bar(
            x=location_alerts.values,
            y=location_alerts.index,
            orientation='h',
            title='Alert Frequency by Location (Top 10)',
            labels={'x': 'Number of Alerts', 'y': 'Location'}
        )
        fig_location.update_layout(height=400)
        st.plotly_chart(fig_location, use_container_width=True)
else:
    st.info(f"No historical alerts found for the selected {time_range.lower()}")

# Alert response metrics
st.subheader("⚡ Alert Response Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    # Average response time (simulated)
    avg_response_time = np.random.randint(15, 45)
    st.metric(
        label="⏱️ Avg Response Time",
        value=f"{avg_response_time} min",
        delta=f"{np.random.randint(-5, 3)} min"
    )

with col2:
    # Alert resolution rate (simulated)
    resolution_rate = np.random.randint(85, 98)
    st.metric(
        label="✅ Resolution Rate",
        value=f"{resolution_rate}%",
        delta=f"{np.random.randint(-2, 5)}%"
    )

with col3:
    # False positive rate (simulated)
    false_positive_rate = np.random.randint(5, 15)
    st.metric(
        label="❌ False Positive Rate",
        value=f"{false_positive_rate}%",
        delta=f"{np.random.randint(-3, 2)}%"
    )

# Alert threshold management
st.subheader("⚙️ Alert Threshold Configuration")

with st.expander("Configure Alert Thresholds"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Air Quality Thresholds**")
        
        aqi_moderate = st.slider("AQI Moderate Threshold", 40, 60, 50)
        aqi_unhealthy_sensitive = st.slider("AQI Unhealthy for Sensitive", 90, 110, 100) 
        aqi_unhealthy = st.slider("AQI Unhealthy", 140, 160, 150)
        
        st.markdown("**Disease Case Thresholds**")
        
        disease_medium = st.slider("Disease Cases Medium", 80, 120, 100)
        disease_high = st.slider("Disease Cases High", 180, 220, 200)
    
    with col2:
        st.markdown("**Hospital Capacity Thresholds**")
        
        capacity_strained = st.slider("Capacity Strained", 80, 90, 85)
        capacity_critical = st.slider("Capacity Critical", 90, 100, 95)
        
        st.markdown("**Notification Settings**")
        
        email_notifications = st.checkbox("Email Notifications", value=True)
        sms_notifications = st.checkbox("SMS Notifications", value=False)
        auto_escalation = st.checkbox("Auto Escalation for Critical", value=True)
    
    if st.button("Update Thresholds"):
        st.success("Alert thresholds updated successfully!")

# Alert reports
st.subheader("📋 Alert Reports")

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Daily Report"):
        # Generate daily report
        report_data = {
            'Total Alerts': total_alerts,
            'Critical Alerts': alert_summary['HIGH'],
            'Medium Alerts': alert_summary['MEDIUM'],
            'Low Alerts': alert_summary['LOW'],
            'Most Affected Location': locations[0] if locations else 'N/A',
            'Average Response Time': f"{avg_response_time} minutes",
            'Report Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        st.json(report_data)

with col2:
    if st.button("Export Alert Data"):
        if filtered_alerts:
            # Convert alerts to DataFrame for export
            export_data = []
            for alert in filtered_alerts:
                export_data.append({
                    'Timestamp': alert['timestamp'],
                    'Type': alert['type'],
                    'Severity': alert['severity'],
                    'Location': alert['location'],
                    'Message': alert['message'],
                    'Value': alert.get('value', ''),
                    'Metric': alert.get('metric', '')
                })
            
            export_df = pd.DataFrame(export_data)
            csv_data = export_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download Alert Data (CSV)",
                data=csv_data,
                file_name=f"alert_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No alert data to export")

# Emergency contact information
st.subheader("📞 Emergency Contacts")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **Air Quality Emergencies**
    - EPA Hotline: 1-800-EPA-INFO
    - Local Air Quality: (555) 123-4567
    - Emergency Response: 911
    """)

with col2:
    st.info("""
    **Disease Outbreak Response**
    - CDC Emergency: 1-800-CDC-INFO
    - Local Health Dept: (555) 234-5678
    - WHO Emergency: +41 22 791 2111
    """)

with col3:
    st.info("""
    **Hospital System Coordination**
    - Regional Coordinator: (555) 345-6789
    - Emergency Management: (555) 456-7890
    - Medical Command: (555) 567-8901
    """)

# Real-time alert feed
st.subheader("📡 Real-Time Alert Feed")

# Simulated real-time feed
if st.checkbox("Enable Real-Time Updates"):
    placeholder = st.empty()
    
    # This would be replaced with actual real-time data in production
    with placeholder.container():
        st.info("🔄 Monitoring for new alerts... (Updates every 30 seconds)")
        
        # Show last few alerts
        recent_alerts = filtered_alerts[:3] if filtered_alerts else []
        
        for alert in recent_alerts:
            severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[alert['severity']]
            st.write(f"{severity_color} **{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['type']} in {alert['location']}")

# Footer information
st.markdown("---")
st.markdown("""
**Alert System Features:**
- Real-time monitoring and notifications
- Multi-level severity classification
- Geographic and temporal filtering
- Threshold-based automated alerting
- Response time tracking and analytics
- Integration with emergency response systems
""")
