"""
API endpoints for data ingestion and external integrations
Provides REST-like interface for triggering data updates
"""
import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from realtime_data_ingestion import data_ingester, scheduled_ingester
from database_service import db_service
from database_models import get_db
import json

class DataAPI:
    """API class for data operations"""
    
    @staticmethod
    def trigger_data_refresh() -> Dict[str, Any]:
        """Trigger a manual data refresh from all sources"""
        try:
            results = data_ingester.ingest_all_sources()
            
            return {
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat(),
                'results': results
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def get_data_freshness() -> Dict[str, Any]:
        """Check how fresh the current data is"""
        try:
            db = next(get_db())
            
            # Get latest data timestamps for each metric type
            freshness_data = {}
            metric_types = ['aqi', 'disease_cases', 'bed_occupancy', 'temperature']
            
            for metric_type in metric_types:
                latest_data = db_service.health_data.get_health_metrics(
                    db, metric_type=metric_type, limit=1
                )
                
                if not latest_data.empty:
                    latest_timestamp = latest_data.iloc[0]['timestamp']
                    age_minutes = (datetime.utcnow() - latest_timestamp).total_seconds() / 60
                    
                    freshness_data[metric_type] = {
                        'latest_timestamp': latest_timestamp.isoformat(),
                        'age_minutes': round(age_minutes, 1),
                        'status': 'fresh' if age_minutes < 60 else 'stale' if age_minutes < 240 else 'very_stale'
                    }
                else:
                    freshness_data[metric_type] = {
                        'status': 'no_data'
                    }
            
            db.close()
            
            return {
                'status': 'success',
                'freshness': freshness_data,
                'checked_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @staticmethod
    def get_api_status() -> Dict[str, Any]:
        """Get status of external API connections"""
        return {
            'status': 'operational',
            'apis': {
                'EPA AirNow': {'status': 'simulated', 'last_success': datetime.utcnow().isoformat()},
                'CDC NNDSS': {'status': 'simulated', 'last_success': datetime.utcnow().isoformat()},
                'HHS Hospital': {'status': 'simulated', 'last_success': datetime.utcnow().isoformat()},
                'NOAA Weather': {'status': 'simulated', 'last_success': datetime.utcnow().isoformat()}
            },
            'note': 'Using simulated data for demonstration. Production deployment would use real API keys.'
        }

def show_data_management_panel():
    """Show data management panel for admins"""
    st.subheader("🔧 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Refresh Controls**")
        
        if st.button("🔄 Refresh All Data Sources", type="primary"):
            with st.spinner("Refreshing data from all sources..."):
                result = DataAPI.trigger_data_refresh()
                
                if result['status'] == 'success':
                    st.success("Data refresh completed!")
                    
                    # Show results summary
                    st.markdown("**Refresh Results:**")
                    for source, data in result['results'].items():
                        status_icon = "✅" if data.get('status') == 'success' else "❌" if data.get('status') == 'error' else "⏭️"
                        records = data.get('records_inserted', 0)
                        st.write(f"{status_icon} {source}: {records} records")
                else:
                    st.error(f"Data refresh failed: {result.get('error', 'Unknown error')}")
        
        if st.button("🌤️ Refresh Air Quality Only"):
            with st.spinner("Refreshing air quality data..."):
                result = scheduled_ingester.run_air_quality_ingestion()
                
                if result['status'] == 'success':
                    st.success(f"Air quality data refreshed: {result['records_inserted']} records")
                else:
                    st.error(f"Air quality refresh failed: {result.get('error')}")
    
    with col2:
        st.markdown("**Data Status**")
        
        # Show data freshness
        if st.button("🕐 Check Data Freshness"):
            freshness = DataAPI.get_data_freshness()
            
            if freshness['status'] == 'success':
                st.markdown("**Data Freshness Status:**")
                for metric, info in freshness['freshness'].items():
                    if info['status'] == 'fresh':
                        st.success(f"{metric}: Fresh ({info.get('age_minutes', 0):.1f} min old)")
                    elif info['status'] == 'stale':
                        st.warning(f"{metric}: Stale ({info.get('age_minutes', 0):.1f} min old)")
                    elif info['status'] == 'very_stale':
                        st.error(f"{metric}: Very Stale ({info.get('age_minutes', 0):.1f} min old)")
                    else:
                        st.info(f"{metric}: No data available")
        
        # Show API status
        if st.button("📡 Check API Status"):
            status = DataAPI.get_api_status()
            
            st.markdown("**External API Status:**")
            for api_name, api_info in status['apis'].items():
                status_color = "🟢" if api_info['status'] == 'operational' else "🟡" if api_info['status'] == 'simulated' else "🔴"
                st.write(f"{status_color} {api_name}: {api_info['status']}")
            
            if status.get('note'):
                st.info(status['note'])

def show_data_sources_info():
    """Show information about configured data sources"""
    st.subheader("📊 Data Sources")
    
    data_sources_info = [
        {
            'name': 'EPA AirNow API',
            'type': 'Air Quality',
            'frequency': 'Hourly',
            'metrics': ['AQI', 'PM2.5', 'PM10', 'Ozone', 'NO2', 'SO2', 'CO'],
            'status': 'Simulated'
        },
        {
            'name': 'CDC NNDSS',
            'type': 'Disease Surveillance',
            'frequency': 'Daily',
            'metrics': ['Disease Cases', 'Outbreak Alerts'],
            'status': 'Simulated'
        },
        {
            'name': 'HHS Hospital Capacity',
            'type': 'Healthcare',
            'frequency': 'Every 2 hours',
            'metrics': ['Bed Occupancy', 'ICU Capacity', 'Emergency Visits', 'Staff Availability'],
            'status': 'Simulated'
        },
        {
            'name': 'NOAA Weather',
            'type': 'Environmental',
            'frequency': 'Hourly',
            'metrics': ['Temperature', 'Humidity', 'Wind Speed', 'Pressure'],
            'status': 'Simulated'
        }
    ]
    
    for source in data_sources_info:
        with st.expander(f"📡 {source['name']} ({source['status']})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {source['type']}")
                st.write(f"**Update Frequency:** {source['frequency']}")
                st.write(f"**Status:** {source['status']}")
            
            with col2:
                st.write("**Metrics:**")
                for metric in source['metrics']:
                    st.write(f"• {metric}")

# Global API instance
data_api = DataAPI()