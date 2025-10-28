"""
Database-backed data sources for the Public Health Monitoring Platform
Replaces mock data with real database data while maintaining the same interface
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from database_service import db_service
from database_models import get_db, Location, HealthMetric, AlertType, AlertSeverity
import uuid

class DatabaseDataSources:
    """Database-backed data sources replacing the mock DataSources"""
    
    def __init__(self):
        self.db = None
        self._seed_sample_data()
    
    def _get_db(self):
        """Get database session"""
        if self.db is None:
            self.db = next(get_db())
        return self.db
    
    def _seed_sample_data(self):
        """Seed the database with sample health data if empty"""
        db = self._get_db()
        
        try:
            # Check if we already have health metrics
            if db.query(HealthMetric).first():
                return
            
            # Get all locations
            locations = db.query(Location).all()
            if not locations:
                return
            
            # Generate sample health metrics for the last 30 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            # Create hourly data points
            current_time = start_date
            metrics_data = []
            
            while current_time <= end_date:
                for location in locations:
                    location_seed = hash(location.name) % 1000
                    np.random.seed(int((location_seed + current_time.timestamp()) % 1000))
                    
                    # Generate realistic health metrics
                    hour = current_time.hour
                    day_of_week = current_time.weekday()
                    
                    # Air Quality Index (with daily patterns)
                    base_aqi = 50 + 30 * np.sin(hour * 2 * np.pi / 24)  # Daily cycle
                    aqi_noise = np.random.normal(0, 10)
                    aqi = max(10, min(300, base_aqi + aqi_noise))
                    
                    # Related air quality metrics
                    pm25 = aqi * 0.6 + np.random.normal(0, 3)
                    pm10 = aqi * 0.8 + np.random.normal(0, 4)
                    ozone = aqi * 0.4 + np.random.normal(0, 2)
                    no2 = aqi * 0.3 + np.random.normal(0, 1.5)
                    so2 = aqi * 0.2 + np.random.normal(0, 1)
                    co = aqi * 0.1 + np.random.normal(0, 0.5)
                    
                    air_quality_metrics = [
                        {'metric_type': 'aqi', 'value': aqi, 'unit': 'AQI'},
                        {'metric_type': 'pm25', 'value': max(0, pm25), 'unit': 'μg/m³'},
                        {'metric_type': 'pm10', 'value': max(0, pm10), 'unit': 'μg/m³'},
                        {'metric_type': 'ozone', 'value': max(0, ozone), 'unit': 'ppb'},
                        {'metric_type': 'no2', 'value': max(0, no2), 'unit': 'ppb'},
                        {'metric_type': 'so2', 'value': max(0, so2), 'unit': 'ppb'},
                        {'metric_type': 'co', 'value': max(0, co), 'unit': 'ppm'}
                    ]
                    
                    # Disease cases (daily, not hourly)
                    if hour == 12:  # Only generate once per day
                        # Weekly pattern + some randomness
                        base_cases = 50 + 20 * np.sin(day_of_week * 2 * np.pi / 7)
                        outbreak_factor = 1 + 0.3 * np.random.random() if np.random.random() < 0.1 else 1
                        disease_cases = max(0, int(base_cases * outbreak_factor + np.random.poisson(10)))
                        
                        air_quality_metrics.append({
                            'metric_type': 'disease_cases', 
                            'value': disease_cases, 
                            'unit': 'cases'
                        })
                    
                    # Hospital capacity (daily)
                    if hour == 12:
                        base_capacity = 75 + 10 * np.sin(day_of_week * 2 * np.pi / 7)
                        capacity_noise = np.random.normal(0, 5)
                        bed_occupancy = max(40, min(98, base_capacity + capacity_noise))
                        
                        icu_occupancy = bed_occupancy * 0.75 + np.random.normal(0, 3)
                        emergency_visits = 100 + bed_occupancy * 2 + np.random.normal(0, 10)
                        available_beds = int((100 - bed_occupancy) * 2.5)
                        staff_availability = max(50, 100 - bed_occupancy * 0.4)
                        
                        hospital_metrics = [
                            {'metric_type': 'bed_occupancy', 'value': bed_occupancy, 'unit': '%'},
                            {'metric_type': 'icu_occupancy', 'value': max(0, min(100, icu_occupancy)), 'unit': '%'},
                            {'metric_type': 'emergency_visits', 'value': max(0, emergency_visits), 'unit': 'visits'},
                            {'metric_type': 'available_beds', 'value': available_beds, 'unit': 'beds'},
                            {'metric_type': 'staff_availability', 'value': max(50, min(100, staff_availability)), 'unit': '%'}
                        ]
                        
                        air_quality_metrics.extend(hospital_metrics)
                    
                    # Add all metrics to the list
                    for metric in air_quality_metrics:
                        metrics_data.append({
                            'id': uuid.uuid4(),
                            'location_id': location.id,
                            'timestamp': current_time,
                            'metric_type': metric['metric_type'],
                            'value': round(metric['value'], 2),
                            'unit': metric['unit'],
                            'source': 'simulation',
                            'quality_score': 0.95
                        })
                
                current_time += timedelta(hours=1)
            
            # Bulk insert metrics
            if metrics_data:
                print(f"Seeding {len(metrics_data)} health metrics...")
                health_metrics = [HealthMetric(**data) for data in metrics_data]
                db.add_all(health_metrics)
                db.commit()
                print("Sample health data seeded successfully!")
                
        except Exception as e:
            print(f"Error seeding sample data: {e}")
            db.rollback()
    
    def get_available_locations(self) -> List[str]:
        """Return list of available location names"""
        db = self._get_db()
        locations = db.query(Location).filter(Location.is_active == True).all()
        return [location.name for location in locations]
    
    def get_current_metrics(self, location: str = "All Locations") -> Dict[str, Any]:
        """Get current metrics for dashboard overview"""
        db = self._get_db()
        
        # Get location ID if specific location selected
        location_id = None
        if location != "All Locations":
            location_obj = db.query(Location).filter(Location.name == location).first()
            if location_obj:
                location_id = str(location_obj.id)
        
        # Use database service to get current metrics
        current_data = db_service.health_data.get_current_metrics(db, location_id)
        
        return current_data
    
    def get_air_quality_data(self, location: str, date_range: Tuple) -> pd.DataFrame:
        """Generate air quality data for specified location and date range"""
        db = self._get_db()
        
        # Convert date range to datetime
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
        
        # Get location ID
        location_id = None
        if location != "All Locations":
            location_obj = db.query(Location).filter(Location.name == location).first()
            if location_obj:
                location_id = str(location_obj.id)
        
        # Get air quality metrics from database
        air_quality_metrics = ['aqi', 'pm25', 'pm10', 'ozone', 'no2', 'so2', 'co']
        
        all_data = []
        for metric_type in air_quality_metrics:
            df = db_service.health_data.get_health_metrics(
                db, 
                location_id=location_id,
                metric_type=metric_type,
                start_date=start_date,
                end_date=end_date,
                limit=2000
            )
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all metrics
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Pivot to get metrics as columns
        pivot_df = combined_df.pivot_table(
            index=['timestamp', 'location_name'], 
            columns='metric_type', 
            values='value', 
            aggfunc='first'
        ).reset_index()
        
        # Flatten column names
        pivot_df.columns.name = None
        
        # Ensure we have the required columns
        for col in air_quality_metrics:
            if col not in pivot_df.columns:
                pivot_df[col] = 0
        
        # Sort by timestamp
        pivot_df = pivot_df.sort_values('timestamp').reset_index(drop=True)
        
        return pivot_df
    
    def get_disease_data(self, location: str, date_range: Tuple) -> pd.DataFrame:
        """Generate disease outbreak data"""
        db = self._get_db()
        
        # Convert date range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
        
        # Get location ID
        location_id = None
        if location != "All Locations":
            location_obj = db.query(Location).filter(Location.name == location).first()
            if location_obj:
                location_id = str(location_obj.id)
        
        # Get disease case data
        df = db_service.health_data.get_health_metrics(
            db,
            location_id=location_id,
            metric_type='disease_cases',
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Rename columns to match expected interface
        result_df = df.rename(columns={'value': 'cases'})
        
        # Group by day if we have hourly data
        result_df['date'] = pd.to_datetime(result_df['timestamp']).dt.date
        daily_df = result_df.groupby('date').agg({
            'cases': 'max',  # Take max cases per day
        }).reset_index()
        
        daily_df['timestamp'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.drop('date', axis=1)
        
        return daily_df
    
    def get_hospital_capacity_data(self, location: str, date_range: Tuple) -> pd.DataFrame:
        """Generate hospital capacity data"""
        db = self._get_db()
        
        # Convert date range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
        
        # Get location ID
        location_id = None
        if location != "All Locations":
            location_obj = db.query(Location).filter(Location.name == location).first()
            if location_obj:
                location_id = str(location_obj.id)
        
        # Get hospital capacity metrics
        capacity_metrics = ['bed_occupancy', 'icu_occupancy', 'emergency_visits', 'available_beds', 'staff_availability']
        
        all_data = []
        for metric_type in capacity_metrics:
            df = db_service.health_data.get_health_metrics(
                db,
                location_id=location_id,
                metric_type=metric_type,
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all metrics
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Pivot to get metrics as columns
        pivot_df = combined_df.pivot_table(
            index=['timestamp', 'location_name'], 
            columns='metric_type', 
            values='value', 
            aggfunc='first'
        ).reset_index()
        
        # Flatten column names
        pivot_df.columns.name = None
        
        # Ensure we have required columns
        for col in capacity_metrics:
            if col not in pivot_df.columns:
                pivot_df[col] = 75  # Default capacity
        
        # Sort by timestamp
        pivot_df = pivot_df.sort_values('timestamp').reset_index(drop=True)
        
        return pivot_df
    
    def get_geospatial_data(self) -> pd.DataFrame:
        """Generate geospatial data for mapping"""
        db = self._get_db()
        
        # Get all locations with their latest metrics
        locations = db.query(Location).filter(Location.is_active == True).all()
        
        geo_data = []
        for location in locations:
            # Get current metrics for this location
            current_metrics = db_service.health_data.get_current_metrics(
                db, str(location.id)
            )
            
            geo_data.append({
                'location': location.name,
                'latitude': location.latitude,
                'longitude': location.longitude,
                'population': location.population or 1000000,
                'aqi': current_metrics.get('aqi', 50),
                'disease_cases': current_metrics.get('disease_cases', 100),
                'hospital_capacity': current_metrics.get('hospital_capacity', 75),
                'risk_level': current_metrics.get('risk_level', 'LOW')
            })
        
        return pd.DataFrame(geo_data)
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()
            self.db = None