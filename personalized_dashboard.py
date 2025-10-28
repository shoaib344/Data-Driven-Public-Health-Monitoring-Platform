"""
Personalized dashboard components for Public Health Monitor
Provides user-specific views, preferences, and customizations
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from database_service import db_service
from database_models import get_db, UserRole, AlertType, AlertSeverity
from auth import get_current_user_id, get_current_user, has_role
from visualizations import Visualizations

class PersonalizedDashboard:
    """Main class for personalized dashboard functionality"""
    
    def __init__(self):
        self.viz = Visualizations()
    
    @property
    def user_id(self):
        """Get current user ID (lazy loaded)"""
        return get_current_user_id()
    
    @property  
    def user(self):
        """Get current user (lazy loaded)"""
        return get_current_user()
    
    def show_personalized_overview(self, data_sources, alert_system):
        """Show personalized dashboard overview"""
        if not self.user:
            st.error("User authentication required")
            return
        
        st.subheader(f"👋 Welcome back, {self.user.get('first_name', self.user['username'])}!")
        
        # Get user preferences
        db = next(get_db())
        try:
            user_locations = db_service.preferences.get_user_locations(db, self.user_id)
            user_alert_prefs = db_service.preferences.get_user_alert_preferences(db, self.user_id)
            
            # Show user's preferred locations
            if user_locations:
                st.markdown("**📍 Your Locations**")
                location_names = [loc.name for loc in user_locations]
                
                # Create tabs for each user location
                if len(location_names) == 1:
                    self._show_location_overview(data_sources, alert_system, location_names[0])
                else:
                    location_tabs = st.tabs(location_names[:4])  # Limit to 4 tabs
                    
                    for i, location_name in enumerate(location_names[:4]):
                        with location_tabs[i]:
                            self._show_location_overview(data_sources, alert_system, location_name)
            else:
                st.info("🔧 **Setup Required**: Please add your preferred locations in Profile & Settings to see personalized data.")
                
                # Quick location selector
                st.markdown("**Quick Setup:**")
                all_locations = data_sources.get_available_locations()
                quick_location = st.selectbox(
                    "Select a location to add to your preferences:",
                    options=[None] + all_locations,
                    format_func=lambda x: "Choose a location..." if x is None else x
                )
                
                if quick_location and st.button("Add to My Locations"):
                    self._add_user_location(quick_location)
                    st.success(f"Added {quick_location} to your locations!")
                    st.rerun()
            
            # Show personalized alerts summary
            self._show_personalized_alerts_summary(alert_system, user_alert_prefs)
            
        finally:
            db.close()
    
    def _show_location_overview(self, data_sources, alert_system, location_name):
        """Show overview for a specific user location"""
        # Get current metrics for this location
        current_data = data_sources.get_current_metrics(location_name)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            aqi_color = "normal" if current_data['aqi'] < 50 else "inverse"
            st.metric(
                label="Air Quality Index",
                value=int(current_data['aqi']),
                delta=current_data.get('aqi_change', 0),
                delta_color=aqi_color
            )
        
        with col2:
            st.metric(
                label="Disease Cases",
                value=int(current_data['disease_cases']),
                delta=current_data.get('disease_change', 0)
            )
        
        with col3:
            capacity_color = "inverse" if current_data['hospital_capacity'] > 85 else "normal"
            st.metric(
                label="Hospital Capacity",
                value=f"{current_data['hospital_capacity']:.0f}%",
                delta=current_data.get('capacity_change', 0),
                delta_color=capacity_color
            )
        
        with col4:
            risk_color = "inverse" if current_data['risk_level'] == "HIGH" else "normal"
            st.metric(
                label="Risk Level",
                value=current_data['risk_level'],
                delta_color=risk_color
            )
        
        # Location-specific alerts
        location_alerts = alert_system.get_current_alerts(location=location_name, user_id=self.user_id)
        if location_alerts:
            st.markdown("**🚨 Location Alerts**")
            for alert in location_alerts[:3]:  # Show top 3 alerts
                severity_icon = "🔴" if alert['severity'] == 'HIGH' else "🟡" if alert['severity'] == 'MEDIUM' else "🟢"
                st.write(f"{severity_icon} {alert['message']}")
        
        # Quick trend chart
        date_range = (datetime.now() - timedelta(days=7), datetime.now())
        air_data = data_sources.get_air_quality_data(location_name, date_range)
        
        if not air_data.empty:
            # Create mini trend chart
            fig = px.line(
                air_data, 
                x='timestamp', 
                y='aqi',
                title=f"7-Day AQI Trend - {location_name}",
                height=300
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_personalized_alerts_summary(self, alert_system, user_alert_prefs):
        """Show summary of user's alert preferences and active alerts"""
        st.markdown("---")
        st.subheader("🔔 Your Alerts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Alert Preferences**")
            if user_alert_prefs:
                for pref in user_alert_prefs:
                    channels = []
                    if pref.email_enabled:
                        channels.append("📧")
                    if pref.sms_enabled:
                        channels.append("📱")
                    if pref.push_enabled:
                        channels.append("🔔")
                    
                    channels_str = " ".join(channels) if channels else "❌"
                    st.write(f"• {pref.alert_type.value}: {pref.severity_threshold.value}+ {channels_str}")
            else:
                st.info("No alert preferences configured. Set them up in Profile & Settings.")
        
        with col2:
            st.markdown("**Recent Alerts**")
            recent_alerts = alert_system.get_current_alerts(location="All Locations", user_id=self.user_id)
            if recent_alerts:
                for alert in recent_alerts[:5]:
                    age = (datetime.now() - alert['timestamp']).total_seconds() / 3600
                    age_str = f"{age:.0f}h ago" if age >= 1 else f"{age*60:.0f}m ago"
                    severity_icon = "🔴" if alert['severity'] == 'HIGH' else "🟡" if alert['severity'] == 'MEDIUM' else "🟢"
                    st.write(f"{severity_icon} {alert['type']} - {age_str}")
            else:
                st.success("No recent alerts")
    
    def show_role_specific_content(self, data_sources, alert_system):
        """Show content based on user role"""
        user_role = self.user['role']
        
        if user_role == UserRole.ADMIN.value:
            self._show_admin_dashboard(data_sources, alert_system)
        elif user_role == UserRole.HEALTH_AUTHORITY.value:
            self._show_health_authority_dashboard(data_sources, alert_system)
        else:
            self._show_public_user_dashboard(data_sources, alert_system)
    
    def _show_admin_dashboard(self, data_sources, alert_system):
        """Admin-specific dashboard content"""
        st.markdown("---")
        st.subheader("👑 Administrator Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**System Status**")
            # System metrics
            db = next(get_db())
            try:
                from database_models import User, HealthMetric, Alert
                
                total_users = db.query(User).count()
                total_metrics = db.query(HealthMetric).count()
                active_alerts = db.query(Alert).filter(Alert.is_active == True).count()
                
                st.metric("Total Users", total_users)
                st.metric("Total Metrics", total_metrics)
                st.metric("Active Alerts", active_alerts)
            finally:
                db.close()
        
        with col2:
            st.markdown("**Alert Summary**")
            alert_summary = alert_system.get_alert_summary()
            st.metric("🔴 High Severity", alert_summary.get('HIGH', 0))
            st.metric("🟡 Medium Severity", alert_summary.get('MEDIUM', 0))
            st.metric("🟢 Low Severity", alert_summary.get('LOW', 0))
        
        with col3:
            st.markdown("**Quick Actions**")
            if st.button("📊 System Analytics"):
                st.info("Detailed system analytics coming soon!")
            if st.button("👥 User Management"):
                st.info("User management interface coming soon!")
            if st.button("⚙️ System Settings"):
                st.info("System settings interface coming soon!")
    
    def _show_health_authority_dashboard(self, data_sources, alert_system):
        """Health authority specific dashboard content"""
        st.markdown("---")
        st.subheader("🏥 Health Authority Dashboard")
        
        # Regional data analysis
        st.markdown("**Regional Health Overview**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Disease trend analysis
            st.markdown("**Disease Surveillance**")
            
            # Get regional disease data
            locations = data_sources.get_available_locations()
            regional_data = []
            
            for location in locations[:5]:  # Top 5 locations
                current_metrics = data_sources.get_current_metrics(location)
                regional_data.append({
                    'Location': location,
                    'Disease Cases': current_metrics['disease_cases'],
                    'Risk Level': current_metrics['risk_level']
                })
            
            if regional_data:
                df = pd.DataFrame(regional_data)
                st.dataframe(df, use_container_width=True)
        
        with col2:
            # Hospital capacity monitoring
            st.markdown("**Hospital Capacity Monitoring**")
            
            capacity_data = []
            for location in locations[:5]:
                current_metrics = data_sources.get_current_metrics(location)
                capacity_data.append({
                    'Location': location,
                    'Capacity %': current_metrics['hospital_capacity'],
                    'Status': 'Critical' if current_metrics['hospital_capacity'] > 90 else 'High' if current_metrics['hospital_capacity'] > 80 else 'Normal'
                })
            
            if capacity_data:
                df = pd.DataFrame(capacity_data)
                
                # Color code the status
                def highlight_status(val):
                    color = 'red' if val == 'Critical' else 'orange' if val == 'High' else 'green'
                    return f'background-color: {color}; color: white'
                
                styled_df = df.style.applymap(highlight_status, subset=['Status'])
                st.dataframe(styled_df, use_container_width=True)
        
        # Authority-specific alerts
        st.markdown("**Regional Alerts**")
        authority_alerts = alert_system.get_current_alerts()
        
        if authority_alerts:
            alert_df = pd.DataFrame([
                {
                    'Location': alert['location'],
                    'Type': alert['type'],
                    'Severity': alert['severity'],
                    'Message': alert['message'][:50] + "..." if len(alert['message']) > 50 else alert['message']
                }
                for alert in authority_alerts[:10]
            ])
            st.dataframe(alert_df, use_container_width=True)
        else:
            st.success("No regional alerts at this time")
    
    def _show_public_user_dashboard(self, data_sources, alert_system):
        """Public user specific dashboard content"""
        st.markdown("---")
        st.subheader("🏠 Your Health Environment")
        
        # Personal health tips based on current conditions
        db = next(get_db())
        try:
            user_locations = db_service.preferences.get_user_locations(db, self.user_id)
            
            if user_locations:
                st.markdown("**Health Recommendations**")
                
                for location in user_locations[:2]:  # Show recommendations for top 2 locations
                    current_metrics = data_sources.get_current_metrics(location.name)
                    recommendations = self._generate_health_recommendations(current_metrics, location.name)
                    
                    with st.expander(f"💡 Health Tips for {location.name}"):
                        for rec in recommendations:
                            st.write(f"• {rec}")
                
                # Personal exposure summary
                st.markdown("**Your Exposure Summary**")
                
                # Calculate aggregate exposure across user locations
                total_aqi = sum(data_sources.get_current_metrics(loc.name)['aqi'] for loc in user_locations)
                avg_aqi = total_aqi / len(user_locations)
                
                total_risk_score = sum(1 if data_sources.get_current_metrics(loc.name)['risk_level'] == 'HIGH' else 0.5 if data_sources.get_current_metrics(loc.name)['risk_level'] == 'MEDIUM' else 0 for loc in user_locations)
                avg_risk = total_risk_score / len(user_locations)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average AQI", f"{avg_aqi:.0f}", help="Average Air Quality Index across your locations")
                with col2:
                    risk_label = "HIGH" if avg_risk > 0.7 else "MEDIUM" if avg_risk > 0.3 else "LOW"
                    st.metric("Overall Risk", risk_label, help="Combined health risk assessment")
            
        finally:
            db.close()
    
    def _generate_health_recommendations(self, metrics: Dict, location: str) -> List[str]:
        """Generate health recommendations based on current metrics"""
        recommendations = []
        
        aqi = metrics['aqi']
        disease_cases = metrics['disease_cases']
        hospital_capacity = metrics['hospital_capacity']
        risk_level = metrics['risk_level']
        
        # Air quality recommendations
        if aqi > 150:
            recommendations.append("🫁 Avoid all outdoor activities. Air quality is very unhealthy.")
            recommendations.append("🏠 Keep windows closed and use air purifiers if available.")
        elif aqi > 100:
            recommendations.append("🚶 Limit prolonged outdoor activities, especially exercise.")
            recommendations.append("😷 Consider wearing an N95 mask when outdoors.")
        elif aqi > 50:
            recommendations.append("🏃 Outdoor activities are generally safe, but sensitive individuals should be cautious.")
        else:
            recommendations.append("🌱 Great air quality! Perfect time for outdoor activities.")
        
        # Disease recommendations
        if disease_cases > 200:
            recommendations.append("🦠 Higher than usual disease activity. Practice extra hygiene precautions.")
            recommendations.append("👥 Consider avoiding crowded areas and maintain social distancing.")
        elif disease_cases > 100:
            recommendations.append("🧼 Wash hands frequently and avoid touching your face.")
        
        # Hospital capacity warnings
        if hospital_capacity > 90:
            recommendations.append("🏥 Hospitals are near capacity. Avoid non-essential medical visits.")
            recommendations.append("🚨 In case of emergency, expect longer wait times.")
        
        # General risk-based advice
        if risk_level == "HIGH":
            recommendations.append("⚠️ Overall risk is elevated. Consider postponing non-essential activities.")
        elif risk_level == "MEDIUM":
            recommendations.append("⚡ Moderate risk conditions. Stay informed and take standard precautions.")
        
        # Always include at least one positive recommendation
        if not recommendations or all("avoid" in rec.lower() or "limit" in rec.lower() for rec in recommendations):
            recommendations.append("💪 Stay hydrated and maintain a healthy lifestyle to boost your immunity.")
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    def _add_user_location(self, location_name: str):
        """Add a location to user preferences"""
        db = next(get_db())
        try:
            # Get location ID
            from database_models import Location
            location = db.query(Location).filter(Location.name == location_name).first()
            
            if location:
                db_service.preferences.add_user_location(
                    db, self.user_id, str(location.id), is_favorite=True
                )
        finally:
            db.close()
    
    def show_personalized_settings(self):
        """Show personalized dashboard settings"""
        st.subheader("⚙️ Dashboard Preferences")
        
        # Dashboard layout preferences
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Display Options**")
            
            show_recommendations = st.checkbox(
                "Show Health Recommendations",
                value=True,
                help="Display personalized health tips based on current conditions"
            )
            
            show_trends = st.checkbox(
                "Show Trend Charts",
                value=True,
                help="Display mini trend charts on the dashboard"
            )
            
            compact_view = st.checkbox(
                "Compact View",
                value=False,
                help="Use a more compact layout with smaller components"
            )
        
        with col2:
            st.markdown("**Update Frequency**")
            
            refresh_interval = st.selectbox(
                "Auto-refresh Dashboard",
                options=[0, 5, 10, 30, 60],
                index=2,
                format_func=lambda x: "Disabled" if x == 0 else f"Every {x} minutes",
                help="How often to automatically refresh dashboard data"
            )
            
            location_limit = st.number_input(
                "Max Locations to Show",
                min_value=1,
                max_value=10,
                value=3,
                help="Maximum number of locations to display on the dashboard"
            )
        
        if st.button("Save Dashboard Preferences"):
            # In a full implementation, these would be saved to the database
            st.session_state.dashboard_preferences = {
                'show_recommendations': show_recommendations,
                'show_trends': show_trends,
                'compact_view': compact_view,
                'refresh_interval': refresh_interval,
                'location_limit': location_limit
            }
            st.success("Dashboard preferences saved!")

# Global personalized dashboard instance
personalized_dashboard = PersonalizedDashboard()