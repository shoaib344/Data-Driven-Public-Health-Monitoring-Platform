from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class AlertSystem:
    def __init__(self):
        self.thresholds = {
            'aqi': {'moderate': 50, 'unhealthy_sensitive': 100, 'unhealthy': 150, 'very_unhealthy': 200},
            'disease_cases': {'low': 50, 'medium': 100, 'high': 200, 'critical': 500},
            'hospital_capacity': {'normal': 70, 'strained': 85, 'critical': 95},
            'pm25': {'good': 12, 'moderate': 35, 'unhealthy_sensitive': 55, 'unhealthy': 150}
        }
        
        self.alert_history = []
        
    def check_air_quality_alerts(self, data: pd.DataFrame, location: str) -> List[Dict]:
        """Check for air quality related alerts"""
        alerts = []
        
        if data.empty:
            return alerts
        
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
            elif aqi_value >= self.thresholds['aqi']['moderate']:
                alerts.append({
                    'type': 'Air Quality',
                    'severity': 'LOW',
                    'message': f'Moderate air quality (AQI: {aqi_value}). Unusually sensitive people should consider limiting prolonged outdoor exertion.',
                    'location': location,
                    'timestamp': datetime.now(),
                    'value': aqi_value,
                    'metric': 'AQI'
                })
        
        # Check for PM2.5 if available
        if 'pm25' in latest_data:
            pm25_value = latest_data['pm25']
            if pm25_value >= self.thresholds['pm25']['unhealthy']:
                alerts.append({
                    'type': 'PM2.5 Pollution',
                    'severity': 'HIGH',
                    'message': f'Dangerous PM2.5 levels detected ({pm25_value} μg/m³). Stay indoors if possible.',
                    'location': location,
                    'timestamp': datetime.now(),
                    'value': pm25_value,
                    'metric': 'PM2.5'
                })
        
        return alerts
    
    def check_disease_alerts(self, data: pd.DataFrame, location: str) -> List[Dict]:
        """Check for disease outbreak alerts"""
        alerts = []
        
        if data.empty:
            return alerts
        
        # Check for recent spike in cases
        if len(data) >= 7:  # Need at least a week of data
            recent_cases = data.tail(3)['cases'].sum()  # Last 3 days
            previous_cases = data.tail(10).head(7)['cases'].sum()  # Previous 7 days
            
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
                elif recent_avg > avg_previous * 1.5:  # 150% increase
                    alerts.append({
                        'type': 'Disease Outbreak',
                        'severity': 'MEDIUM',
                        'message': f'Moderate increase in disease cases. Recent 3-day average: {recent_avg:.0f} vs previous week average: {avg_previous:.0f}',
                        'location': location,
                        'timestamp': datetime.now(),
                        'value': recent_avg,
                        'metric': 'Cases per day'
                    })
        
        # Check absolute thresholds
        latest_cases = data.tail(1).iloc[0]['cases']
        
        if latest_cases >= self.thresholds['disease_cases']['critical']:
            alerts.append({
                'type': 'Disease Cases',
                'severity': 'HIGH',
                'message': f'Critical number of disease cases reported: {latest_cases}. Emergency protocols may be activated.',
                'location': location,
                'timestamp': datetime.now(),
                'value': latest_cases,
                'metric': 'Daily cases'
            })
        elif latest_cases >= self.thresholds['disease_cases']['high']:
            alerts.append({
                'type': 'Disease Cases',
                'severity': 'MEDIUM',
                'message': f'High number of disease cases reported: {latest_cases}. Enhanced monitoring in effect.',
                'location': location,
                'timestamp': datetime.now(),
                'value': latest_cases,
                'metric': 'Daily cases'
            })
        
        return alerts
    
    def check_hospital_capacity_alerts(self, data: pd.DataFrame, location: str) -> List[Dict]:
        """Check for hospital capacity alerts"""
        alerts = []
        
        if data.empty:
            return alerts
        
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
            elif occupancy >= self.thresholds['hospital_capacity']['strained']:
                alerts.append({
                    'type': 'Hospital Capacity',
                    'severity': 'MEDIUM',
                    'message': f'Hospital system under strain: {occupancy}% bed occupancy. Non-urgent procedures may be delayed.',
                    'location': location,
                    'timestamp': datetime.now(),
                    'value': occupancy,
                    'metric': 'Bed occupancy %'
                })
        
        # Check ICU capacity if available
        if 'icu_occupancy' in latest_data:
            icu_occupancy = latest_data['icu_occupancy']
            
            if icu_occupancy >= 90:
                alerts.append({
                    'type': 'ICU Capacity',
                    'severity': 'HIGH',
                    'message': f'Critical ICU occupancy: {icu_occupancy}%. Patient transfers may be necessary.',
                    'location': location,
                    'timestamp': datetime.now(),
                    'value': icu_occupancy,
                    'metric': 'ICU occupancy %'
                })
        
        return alerts
    
    def check_trend_alerts(self, data: pd.DataFrame, location: str, metric: str) -> List[Dict]:
        """Check for concerning trends in data"""
        alerts = []
        
        if data.empty or len(data) < 5:
            return alerts
        
        # Calculate trend over last 5 data points
        recent_data = data.tail(5)[metric]
        
        if recent_data.isna().all():
            return alerts
        
        # Simple trend calculation
        x = np.arange(len(recent_data))
        coefficients = np.polyfit(x, recent_data, 1)
        trend_slope = coefficients[0]
        
        # Determine if trend is concerning
        if metric == 'aqi' and trend_slope > 5:  # AQI increasing by more than 5 per time period
            alerts.append({
                'type': 'Trend Alert',
                'severity': 'MEDIUM',
                'message': f'Air quality deteriorating rapidly in {location}. AQI trend: +{trend_slope:.1f} per period.',
                'location': location,
                'timestamp': datetime.now(),
                'value': trend_slope,
                'metric': f'{metric} trend'
            })
        elif metric == 'cases' and trend_slope > 10:  # Disease cases increasing rapidly
            alerts.append({
                'type': 'Trend Alert',
                'severity': 'MEDIUM',
                'message': f'Disease cases increasing rapidly in {location}. Trend: +{trend_slope:.1f} cases per period.',
                'location': location,
                'timestamp': datetime.now(),
                'value': trend_slope,
                'metric': f'{metric} trend'
            })
        elif metric == 'bed_occupancy' and trend_slope > 3:  # Hospital capacity filling up
            alerts.append({
                'type': 'Trend Alert',
                'severity': 'MEDIUM',
                'message': f'Hospital capacity increasing rapidly in {location}. Trend: +{trend_slope:.1f}% per period.',
                'location': location,
                'timestamp': datetime.now(),
                'value': trend_slope,
                'metric': f'{metric} trend'
            })
        
        return alerts
    
    def get_current_alerts(self, location: str = "All Locations") -> List[Dict]:
        """Get all current alerts for a location"""
        # This would typically pull from a real-time data source
        # For demo purposes, we'll generate some sample alerts based on current conditions
        
        all_alerts = []
        
        # Simulate some current alerts
        np.random.seed(hash(location + str(datetime.now().date())) % 1000)
        
        # Random chance of alerts
        if np.random.random() < 0.3:  # 30% chance of air quality alert
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
            
            all_alerts.append({
                'type': 'Air Quality',
                'severity': severity,
                'message': message,
                'location': location,
                'timestamp': datetime.now(),
                'value': aqi_value,
                'metric': 'AQI'
            })
        
        if np.random.random() < 0.2:  # 20% chance of disease alert
            cases = np.random.randint(150, 400)
            severity = 'HIGH' if cases > 300 else 'MEDIUM'
            all_alerts.append({
                'type': 'Disease Outbreak',
                'severity': severity,
                'message': f'Elevated disease activity: {cases} cases reported in the last 24 hours.',
                'location': location,
                'timestamp': datetime.now(),
                'value': cases,
                'metric': 'Daily cases'
            })
        
        if np.random.random() < 0.15:  # 15% chance of hospital capacity alert
            capacity = np.random.randint(85, 98)
            severity = 'HIGH' if capacity > 95 else 'MEDIUM'
            all_alerts.append({
                'type': 'Hospital Capacity',
                'severity': severity,
                'message': f'Hospital system strained: {capacity}% bed occupancy.',
                'location': location,
                'timestamp': datetime.now(),
                'value': capacity,
                'metric': 'Bed occupancy %'
            })
        
        return all_alerts
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alert counts by severity"""
        # This would typically query a database of active alerts
        return {
            'HIGH': np.random.randint(0, 5),
            'MEDIUM': np.random.randint(0, 8),
            'LOW': np.random.randint(0, 12)
        }
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss a specific alert"""
        # In a real system, this would update the database
        return True
    
    def get_alert_history(self, location: str = None, days: int = 7) -> List[Dict]:
        """Get historical alerts"""
        # Generate some sample historical alerts
        history = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            # Random alerts for each day
            if np.random.random() < 0.4:  # 40% chance of alert per day
                alert_types = ['Air Quality', 'Disease Outbreak', 'Hospital Capacity']
                alert_type = np.random.choice(alert_types)
                severity = np.random.choice(['LOW', 'MEDIUM', 'HIGH'], p=[0.5, 0.3, 0.2])
                
                history.append({
                    'type': alert_type,
                    'severity': severity,
                    'message': f'Historical {alert_type.lower()} alert',
                    'location': location or 'Various',
                    'timestamp': date,
                    'resolved': True
                })
        
        return history
