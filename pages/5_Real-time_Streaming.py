import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import check_authentication, show_login_page, has_role, UserRole
from streaming_interface import streaming_interface

st.set_page_config(
    page_title="Real-time Streaming - Public Health Monitor",
    page_icon="📡",
    layout="wide"
)

# Check authentication
if not check_authentication():
    show_login_page()
    st.stop()

st.title("📡 Real-time Streaming Dashboard")
st.markdown("Live data streaming and event broadcasting for instant health updates")

# Show appropriate interface based on user role
if has_role(UserRole.ADMIN):
    # Full streaming dashboard for admins
    streaming_interface.show_streaming_dashboard()
elif has_role(UserRole.HEALTH_AUTHORITY):
    # Limited streaming info for health authorities
    st.subheader("📊 Streaming Status")
    
    streaming_interface.show_streaming_status_widget()
    
    st.markdown("---")
    st.markdown("""
    **Real-time Streaming Features:**
    
    🔄 **Live Data Updates**: Automatic broadcasting of health metric changes to connected users
    
    🚨 **Instant Alerts**: Real-time delivery of health alerts based on user preferences
    
    🤖 **ML Predictions**: Live streaming of new machine learning predictions as they're generated
    
    👥 **User Activity**: Real-time updates of user activity and system status
    
    📍 **Location-based**: Users receive updates only for their subscribed locations
    
    📱 **Multi-device**: WebSocket connections work across web, mobile, and desktop platforms
    """)
    
    if not has_role(UserRole.ADMIN):
        st.info("💡 Contact your system administrator for full streaming management access.")
        
else:
    # Public users see streaming benefits and connection info
    st.subheader("📱 Live Updates Available")
    
    st.markdown("""
    Your Public Health Monitor supports **real-time streaming** for instant updates:
    
    ### 🔄 What You Get:
    - **Live Health Data**: See air quality, disease cases, and hospital capacity update automatically
    - **Instant Alerts**: Get notified immediately when health conditions change in your area  
    - **ML Predictions**: Receive new AI predictions as soon as they're generated
    - **Location Updates**: Auto-refresh data for all your saved locations
    
    ### 📊 How It Works:
    Real-time updates happen automatically when you use the dashboard. No additional setup required!
    
    ### 🌟 Benefits:
    ✅ Always see the latest health information  
    ✅ Get alerts faster than email or SMS  
    ✅ Never miss critical health updates  
    ✅ Seamless experience across all devices  
    """)
    
    # Show current streaming status
    st.markdown("---")
    streaming_interface.show_streaming_status_widget()

# Add some technical information for developers/admins
if has_role(UserRole.ADMIN):
    with st.expander("🔧 Technical Implementation Details"):
        st.markdown("""
        **WebSocket Streaming Architecture:**
        
        - **Server**: WebSocket server running on port 5001 with asyncio
        - **Events**: Health data updates, alerts, ML predictions, system status
        - **Broadcasting**: Location-based and user-specific event routing
        - **Connection Management**: Automatic reconnection, ping/pong heartbeat
        - **Queue Processing**: Asynchronous event processing with 1000-event capacity
        - **Scalability**: Supports concurrent connections with efficient message routing
        
        **Event Types:**
        - `health_data_update`: New health metrics for locations
        - `alert_notification`: User and location alerts
        - `ml_prediction_update`: New AI predictions
        - `system_status`: Server and system status changes
        - `user_activity`: User actions and preferences
        
        **Client Integration:**
        ```javascript
        const ws = new WebSocket('ws://localhost:5001');
        ws.send(JSON.stringify({
            user_id: 'user123',
            locations: ['location1', 'location2'],
            events: ['health_data_update', 'alert_notification']
        }));
        ```
        """)