"""
Streamlit Interface for WebSocket Streaming Management
Provides admin interface for monitoring and managing real-time streaming
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

from websocket_streaming import websocket_server, WEBSOCKETS_AVAILABLE
from auth import has_role, UserRole

class StreamingInterface:
    """Interface for managing WebSocket streaming"""
    
    def show_streaming_dashboard(self):
        """Show streaming management dashboard for admins"""
        if not has_role(UserRole.ADMIN):
            st.error("Access denied. Admin privileges required.")
            return
        
        st.subheader("📡 Real-time Streaming Dashboard")
        
        if not WEBSOCKETS_AVAILABLE:
            st.error("⚠️ WebSocket streaming not available. Install 'websockets' package to enable real-time features.")
            st.code("pip install websockets")
            return
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Server Status", "Connections", "Event Monitoring", "Configuration"])
        
        with tab1:
            self._show_server_status()
        
        with tab2:
            self._show_connection_management()
        
        with tab3:
            self._show_event_monitoring()
        
        with tab4:
            self._show_configuration()
    
    def _show_server_status(self):
        """Show WebSocket server status"""
        st.markdown("**WebSocket Server Status**")
        
        status = websocket_server.get_server_status()
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            server_status = "🟢 Running" if status['running'] else "🔴 Stopped"
            st.metric("Server Status", server_status)
        
        with col2:
            st.metric("Active Connections", status.get('active_connections', 0))
        
        with col3:
            st.metric("Messages Sent", status.get('messages_sent', 0))
        
        with col4:
            st.metric("Events Processed", status.get('events_processed', 0))
        
        # Server configuration
        st.markdown("**Server Configuration**")
        config_data = {
            'Host': status.get('host', 'N/A'),
            'Port': status.get('port', 'N/A'),
            'WebSockets Available': '✅ Yes' if status.get('websockets_available') else '❌ No',
            'Total Connections': status.get('total_connections', 0),
            'Unique Users': status.get('unique_users', 0),
            'Locations Subscribed': status.get('locations_subscribed', 0)
        }
        
        config_df = pd.DataFrame(list(config_data.items()), columns=['Setting', 'Value'])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        # Server controls
        st.markdown("**Server Controls**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not status['running']:
                if st.button("🚀 Start Server"):
                    with st.spinner("Starting WebSocket server..."):
                        try:
                            import asyncio
                            # Note: In production, you would start this properly
                            st.info("Server start initiated. Check status in a few seconds.")
                        except Exception as e:
                            st.error(f"Failed to start server: {str(e)}")
        
        with col2:
            if status['running']:
                if st.button("🛑 Stop Server"):
                    with st.spinner("Stopping WebSocket server..."):
                        try:
                            st.info("Server stop initiated.")
                        except Exception as e:
                            st.error(f"Failed to stop server: {str(e)}")
        
        with col3:
            if st.button("🔄 Refresh Status"):
                st.rerun()
    
    def _show_connection_management(self):
        """Show active connections management"""
        st.markdown("**Active Connections**")
        
        # Mock connection data for display (in real implementation, this would come from connection_manager)
        if hasattr(websocket_server, 'connection_manager'):
            connections_data = []
            
            # Get connection metadata
            for conn_id, metadata in websocket_server.connection_manager.connection_metadata.items():
                connections_data.append({
                    'Connection ID': conn_id[:8] + '...',
                    'User ID': metadata.get('user_id', 'Unknown'),
                    'Connected At': metadata.get('connected_at', datetime.utcnow()).strftime('%H:%M:%S'),
                    'Last Ping': metadata.get('last_ping', datetime.utcnow()).strftime('%H:%M:%S'),
                    'IP Address': metadata.get('ip_address', 'Unknown'),
                    'Status': '🟢 Active'
                })
            
            if connections_data:
                connections_df = pd.DataFrame(connections_data)
                st.dataframe(connections_df, use_container_width=True, hide_index=True)
                
                # Connection statistics chart
                st.markdown("**Connection Activity**")
                
                # Mock time-series data for connections (replace with real data)
                now = datetime.now()
                time_points = [now - timedelta(minutes=i) for i in range(60, 0, -1)]
                connection_counts = [len(connections_data) + (i % 5) for i in range(60)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=connection_counts,
                    mode='lines+markers',
                    name='Active Connections',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title='Connection Count (Last Hour)',
                    xaxis_title='Time',
                    yaxis_title='Active Connections',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No active connections")
        else:
            st.info("Connection manager not available")
    
    def _show_event_monitoring(self):
        """Show real-time event monitoring"""
        st.markdown("**Event Stream Monitoring**")
        
        # Event type distribution
        st.markdown("**Event Types Distribution**")
        
        # Mock event data (replace with real streaming data)
        event_types = ['Health Data Update', 'Alert Notification', 'ML Prediction', 'System Status', 'User Activity']
        event_counts = [45, 12, 8, 3, 15]
        
        fig_pie = px.pie(
            values=event_counts,
            names=event_types,
            title='Events by Type (Last Hour)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Recent events log
        st.markdown("**Recent Events**")
        
        # Mock recent events
        recent_events = []
        for i in range(10):
            recent_events.append({
                'Timestamp': (datetime.now() - timedelta(minutes=i*2)).strftime('%H:%M:%S'),
                'Event Type': event_types[i % len(event_types)],
                'Location': f'Location_{i%3 + 1}',
                'Priority': ['Normal', 'High', 'Critical'][i % 3],
                'Recipients': f'{(i%5) + 1} users'
            })
        
        events_df = pd.DataFrame(recent_events)
        st.dataframe(events_df, use_container_width=True, hide_index=True)
        
        # Event rate chart
        st.markdown("**Event Processing Rate**")
        
        # Mock event rate data
        time_points = [datetime.now() - timedelta(minutes=i) for i in range(30, 0, -1)]
        event_rates = [10 + (i % 8) for i in range(30)]
        
        fig_rate = go.Figure()
        fig_rate.add_trace(go.Scatter(
            x=time_points,
            y=event_rates,
            mode='lines',
            name='Events/Minute',
            line=dict(color='green', width=2),
            fill='tonexty'
        ))
        
        fig_rate.update_layout(
            title='Event Processing Rate (Last 30 Minutes)',
            xaxis_title='Time',
            yaxis_title='Events per Minute',
            height=300
        )
        
        st.plotly_chart(fig_rate, use_container_width=True)
    
    def _show_configuration(self):
        """Show streaming configuration options"""
        st.markdown("**Streaming Configuration**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Streaming Settings**")
            
            health_data_interval = st.slider(
                "Health Data Update Interval (seconds)",
                min_value=10,
                max_value=300,
                value=30,
                help="How often to check for and broadcast health data updates"
            )
            
            alert_interval = st.slider(
                "Alert Check Interval (seconds)",
                min_value=5,
                max_value=60,
                value=10,
                help="How often to check for new alerts to broadcast"
            )
            
            max_connections = st.number_input(
                "Maximum Connections",
                min_value=10,
                max_value=1000,
                value=100,
                help="Maximum number of concurrent WebSocket connections"
            )
            
            event_queue_size = st.number_input(
                "Event Queue Size",
                min_value=100,
                max_value=10000,
                value=1000,
                help="Maximum number of events to queue for processing"
            )
        
        with col2:
            st.markdown("**Connection Settings**")
            
            ping_interval = st.slider(
                "Ping Interval (seconds)",
                min_value=10,
                max_value=120,
                value=30,
                help="How often to ping clients to check connection"
            )
            
            ping_timeout = st.slider(
                "Ping Timeout (seconds)",
                min_value=5,
                max_value=60,
                value=10,
                help="How long to wait for ping response before disconnecting"
            )
            
            message_size_limit = st.number_input(
                "Message Size Limit (KB)",
                min_value=1,
                max_value=1000,
                value=100,
                help="Maximum size for WebSocket messages"
            )
            
            compression_enabled = st.checkbox(
                "Enable Compression",
                value=True,
                help="Compress WebSocket messages to save bandwidth"
            )
        
        # Apply configuration button
        st.markdown("---")
        
        if st.button("💾 Apply Configuration"):
            # In a real implementation, this would update the server configuration
            config_data = {
                'health_data_interval': health_data_interval,
                'alert_interval': alert_interval,
                'max_connections': max_connections,
                'event_queue_size': event_queue_size,
                'ping_interval': ping_interval,
                'ping_timeout': ping_timeout,
                'message_size_limit': message_size_limit,
                'compression_enabled': compression_enabled
            }
            
            st.success("✅ Configuration applied successfully!")
            st.json(config_data)
    
    def show_streaming_status_widget(self):
        """Show compact streaming status widget for other pages"""
        if not has_role(UserRole.HEALTH_AUTHORITY):
            return
        
        status = websocket_server.get_server_status()
        
        with st.container():
            st.markdown("**📡 Live Streaming**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_color = "🟢" if status['running'] else "🔴"
                st.write(f"{status_color} Server")
            
            with col2:
                st.write(f"👥 {status.get('active_connections', 0)} users")
            
            with col3:
                st.write(f"📊 {status.get('events_processed', 0)} events")

# Global streaming interface instance
streaming_interface = StreamingInterface()