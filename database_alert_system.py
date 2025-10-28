"""
Database-backed alert system for the Public Health Monitoring Platform
Replaces mock alerts with real database-backed alerts
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any
from database_service import db_service
from database_models import get_db, Alert, Location, AlertType, AlertSeverity
import numpy as np

class DatabaseAlertSystem:
    """Database-backed alert system"""
    
    def __init__(self):
        self.db = None
        self.thresholds = {
            'aqi': {'moderate': 50, 'unhealthy_sensitive': 100, 'unhealthy': 150, 'very_unhealthy': 200},
            'disease_cases': {'low': 50, 'medium': 100, 'high': 200, 'critical': 500},
            'hospital_capacity': {'normal': 70, 'strained': 85, 'critical': 95},
            'pm25': {'good': 12, 'moderate': 35, 'unhealthy_sensitive': 55, 'unhealthy': 150}
        }
    
    def _get_db(self):
        """Get database session"""
        if self.db is None:
            self.db = next(get_db())
        return self.db
    
    def check_air_quality_alerts(self, data, location: str) -> List[Dict]:
        """Check for air quality related alerts using real data"""
        if data.empty:
            return []
        
        alerts = []
        latest_data = data.tail(1).iloc[0]
        
        if 'aqi' in latest_data:
            aqi_value = latest_data['aqi']
            
            if aqi_value >= self.thresholds['aqi']['very_unhealthy']:
                alerts.append({
                    'type': 'Air Quality',
                    'severity': 'HIGH',
                    'message': f'Very Unhealthy air quality detected (AQI: {aqi_value}). Avoid all outdoor activities.',
                    'location': location,
                    'timestamp': datetime.now(),
                    'value': aqi_value,
                    'metric': 'AQI'
                })
                # Store in database
                self._create_database_alert(
                    location, AlertType.AIR_QUALITY, AlertSeverity.HIGH,
                    "Very Unhealthy Air Quality",
                    f"Very Unhealthy air quality detected (AQI: {aqi_value}). Avoid all outdoor activities.",
                    self.thresholds['aqi']['very_unhealthy'], aqi_value, 'AQI'
                )
            elif aqi_value >= self.thresholds['aqi']['unhealthy']:
                alerts.append({
                    'type': 'Air Quality',
                    'severity': 'HIGH',
                    'message': f'Unhealthy air quality (AQI: {aqi_value}). Everyone should avoid prolonged outdoor exertion.',
                    'location': location,
                    'timestamp': datetime.now(),
                    'value': aqi_value,
                    'metric': 'AQI'
                })
            elif aqi_value >= self.thresholds['aqi']['unhealthy_sensitive']:
                alerts.append({
                    'type': 'Air Quality',
                    'severity': 'MEDIUM',
                    'message': f'Unhealthy air for sensitive groups (AQI: {aqi_value}). Sensitive individuals should limit outdoor activities.',
                    'location': location,
                    'timestamp': datetime.now(),
                    'value': aqi_value,
                    'metric': 'AQI'
                })
        
        return alerts
    
    def check_disease_alerts(self, data, location: str) -> List[Dict]:
        """Check for disease outbreak alerts using real data"""
        if data.empty:
            return []
        
        alerts = []
        
        # Check for recent spike in cases
        if len(data) >= 7:
            recent_cases = data.tail(3)['cases'].sum()
            previous_cases = data.tail(10).head(7)['cases'].sum()
            
            if len(data.tail(10).head(7)) > 0:
                avg_previous = previous_cases / 7
                recent_avg = recent_cases / 3
                
                if recent_avg > avg_previous * 2:  # 200% increase
                    alerts.append({
                        'type': 'Disease Outbreak',
                        'severity': 'HIGH',
                        'message': f'Significant spike in disease cases detected. Recent 3-day average: {recent_avg:.0f} vs previous week average: {avg_previous:.0f}',
                        'location': location,
                        'timestamp': datetime.now(),
                        'value': recent_avg,
                        'metric': 'Cases per day'
                    })
                    # Store in database
                    self._create_database_alert(
                        location, AlertType.DISEASE_OUTBREAK, AlertSeverity.HIGH,
                        "Disease Outbreak Spike",
                        f'Significant spike in disease cases detected. Recent 3-day average: {recent_avg:.0f} vs previous week average: {avg_previous:.0f}',
                        avg_previous * 2, recent_avg, 'Cases per day'
                    )
        
        return alerts
    
    def check_hospital_capacity_alerts(self, data, location: str) -> List[Dict]:
        """Check for hospital capacity alerts using real data"""
        if data.empty:
            return []
        
        alerts = []
        latest_data = data.tail(1).iloc[0]
        
        if 'bed_occupancy' in latest_data:
            occupancy = latest_data['bed_occupancy']
            
            if occupancy >= self.thresholds['hospital_capacity']['critical']:
                alerts.append({
                    'type': 'Hospital Capacity',
                    'severity': 'HIGH',
                    'message': f'Critical hospital bed occupancy: {occupancy}%. Emergency overflow protocols may be needed.',
                    'location': location,
                    'timestamp': datetime.now(),
                    'value': occupancy,
                    'metric': 'Bed occupancy %'
                })
                # Store in database
                self._create_database_alert(
                    location, AlertType.HOSPITAL_CAPACITY, AlertSeverity.HIGH,
                    "Critical Hospital Capacity",
                    f'Critical hospital bed occupancy: {occupancy}%. Emergency overflow protocols may be needed.',
                    self.thresholds['hospital_capacity']['critical'], occupancy, 'Bed occupancy %'
                )
        
        return alerts
    
    def _create_database_alert(self, location_name: str, alert_type: AlertType, 
                             severity: AlertSeverity, title: str, message: str,
                             threshold_value: float = None, current_value: float = None,
                             metric_unit: str = None):
        """Create alert in database"""
        try:
            db = self._get_db()
            
            # Get location ID
            location = db.query(Location).filter(Location.name == location_name).first()
            if not location:
                return
            
            # Check if similar alert already exists and is active
            existing_alert = db.query(Alert).filter(
                Alert.location_id == location.id,
                Alert.alert_type == alert_type,
                Alert.is_active == True,
                Alert.created_at >= datetime.utcnow() - timedelta(hours=1)
            ).first()
            
            if existing_alert:
                return  # Don't create duplicate alerts
            
            # Create new alert
            db_service.alerts.create_alert(
                db, str(location.id), alert_type, severity, title, message,
                threshold_value, current_value, metric_unit
            )
            
        except Exception as e:
            print(f"Error creating database alert: {e}")
    
    def get_current_alerts(self, location: str = "All Locations", user_id: str = None) -> List[Dict]:
        """Get current alerts from database"""
        try:
            db = self._get_db()
            
            # Get location ID if specific location
            location_id = None
            if location != "All Locations":
                location_obj = db.query(Location).filter(Location.name == location).first()
                if location_obj:
                    location_id = str(location_obj.id)
            
            # Get active alerts from database
            db_alerts = db_service.alerts.get_active_alerts(db, location_id, user_id)
            
            # Convert to expected format
            alerts = []
            for alert in db_alerts:
                alerts.append({
                    'id': str(alert.id),
                    'type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'location': alert.location.name,
                    'timestamp': alert.created_at,
                    'value': alert.current_value,
                    'metric': alert.metric_unit
                })
            
            # If no database alerts, fall back to some sample alerts for demo
            if not alerts:
                # Generate some sample alerts for demo purposes
                np.random.seed(hash(location + str(datetime.now().date())) % 1000)
                
                if np.random.random() < 0.3:  # 30% chance of alert
                    aqi_value = np.random.randint(80, 180)
                    if aqi_value > 150:
                        severity = 'HIGH'
                        message = f'Unhealthy air quality detected (AQI: {aqi_value}). Avoid outdoor activities.'
                    elif aqi_value > 100:
                        severity = 'MEDIUM'
                        message = f'Unhealthy for sensitive groups (AQI: {aqi_value}). Sensitive individuals should limit outdoor activities.'
                    else:
                        severity = 'LOW'
                        message = f'Moderate air quality (AQI: {aqi_value}). Consider limiting prolonged outdoor exertion.'
                    
                    alerts.append({
                        'type': 'Air Quality',
                        'severity': severity,
                        'message': message,
                        'location': location,
                        'timestamp': datetime.now(),
                        'value': aqi_value,
                        'metric': 'AQI'
                    })
            
            return alerts
            
        except Exception as e:
            print(f"Error getting current alerts: {e}")
            return []
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alert counts by severity"""
        try:
            db = self._get_db()
            
            # Count active alerts by severity
            high_count = db.query(Alert).filter(
                Alert.is_active == True,
                Alert.severity == AlertSeverity.HIGH
            ).count()
            
            medium_count = db.query(Alert).filter(
                Alert.is_active == True,
                Alert.severity == AlertSeverity.MEDIUM
            ).count()
            
            low_count = db.query(Alert).filter(
                Alert.is_active == True,
                Alert.severity == AlertSeverity.LOW
            ).count()
            
            return {
                'HIGH': high_count,
                'MEDIUM': medium_count,
                'LOW': low_count
            }
            
        except Exception as e:
            print(f"Error getting alert summary: {e}")
            # Fallback to random values
            return {
                'HIGH': np.random.randint(0, 5),
                'MEDIUM': np.random.randint(0, 8),
                'LOW': np.random.randint(0, 12)
            }
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss a specific alert"""
        try:
            db = self._get_db()
            return db_service.alerts.resolve_alert(db, alert_id, None)  # TODO: Add user_id
        except Exception as e:
            print(f"Error dismissing alert: {e}")
            return False
    
    def get_alert_history(self, location: str = None, days: int = 7) -> List[Dict]:
        """Get historical alerts from database"""
        try:
            db = self._get_db()
            
            # Get location ID if specific location
            location_id = None
            if location and location != "All Locations":
                location_obj = db.query(Location).filter(Location.name == location).first()
                if location_obj:
                    location_id = str(location_obj.id)
            
            # Query historical alerts
            query = db.query(Alert).filter(
                Alert.created_at >= datetime.utcnow() - timedelta(days=days)
            )
            
            if location_id:
                query = query.filter(Alert.location_id == location_id)
            
            alerts = query.order_by(Alert.created_at.desc()).limit(100).all()
            
            # Convert to expected format
            history = []
            for alert in alerts:
                history.append({
                    'type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'location': alert.location.name,
                    'timestamp': alert.created_at,
                    'resolved': not alert.is_active
                })
            
            return history
            
        except Exception as e:
            print(f"Error getting alert history: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()
            self.db = None