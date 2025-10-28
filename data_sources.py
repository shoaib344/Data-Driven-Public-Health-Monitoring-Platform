import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any, Tuple

class DataSources:
    def __init__(self):
        self.locations = [
            "New York, NY", "Los Angeles, CA", "Chicago, IL", 
            "Houston, TX", "Phoenix, AZ", "Philadelphia, PA",
            "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA"
        ]
        
        # Initialize base data
        self._initialize_base_data()
    
    def _initialize_base_data(self):
        """Initialize base datasets for the application"""
        # Generate 30 days of historical data
        self.date_range = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='h'
        )
        
    def get_available_locations(self) -> List[str]:
        """Return list of available locations"""
        return self.locations
    
    def get_current_metrics(self, location: str = "All Locations") -> Dict[str, Any]:
        """Get current metrics for dashboard overview"""
        np.random.seed(42)  # For reproducible results
        
        if location == "All Locations":
            # Aggregate metrics
            base_aqi = np.random.randint(50, 150)
            base_cases = np.random.randint(100, 500)
            base_capacity = np.random.randint(70, 95)
        else:
            # Location-specific metrics
            location_seed = hash(location) % 1000
            np.random.seed(location_seed)
            base_aqi = np.random.randint(30, 120)
            base_cases = np.random.randint(50, 300)
            base_capacity = np.random.randint(60, 90)
        
        # Determine risk level
        risk_score = (base_aqi / 150 + base_cases / 500 + base_capacity / 100) / 3
        if risk_score > 0.7:
            risk_level = "HIGH"
        elif risk_score > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'aqi': base_aqi,
            'aqi_change': np.random.randint(-10, 15),
            'disease_cases': base_cases,
            'disease_change': np.random.randint(-20, 30),
            'hospital_capacity': base_capacity,
            'capacity_change': np.random.randint(-5, 10),
            'risk_level': risk_level
        }
    
    def get_air_quality_data(self, location: str, date_range: Tuple) -> pd.DataFrame:
        """Generate air quality data for specified location and date range"""
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            dates = pd.date_range(start=start_date, end=end_date, freq='h')
        else:
            dates = self.date_range[-168:]  # Last 7 days
        
        if location == "All Locations":
            # Aggregate data across all locations
            data_points = []
            for loc in self.locations:
                loc_seed = hash(loc) % 1000
                np.random.seed(loc_seed)
                base_aqi = 50 + 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
                noise = np.random.normal(0, 10, len(dates))
                aqi_values = np.maximum(0, base_aqi + noise)
                
                for i, date in enumerate(dates):
                    data_points.append({
                        'timestamp': date,
                        'location': loc,
                        'aqi': int(aqi_values[i]),
                        'pm25': int(aqi_values[i] * 0.6),
                        'pm10': int(aqi_values[i] * 0.8),
                        'ozone': int(aqi_values[i] * 0.4),
                        'no2': int(aqi_values[i] * 0.3),
                        'so2': int(aqi_values[i] * 0.2),
                        'co': int(aqi_values[i] * 0.1)
                    })
            
            df = pd.DataFrame(data_points)
            # Average across locations for each timestamp
            agg_df = df.groupby('timestamp').agg({
                'aqi': 'mean',
                'pm25': 'mean',
                'pm10': 'mean',
                'ozone': 'mean',
                'no2': 'mean',
                'so2': 'mean',
                'co': 'mean'
            }).reset_index()
            
            return agg_df.round(0).astype({'aqi': int, 'pm25': int, 'pm10': int, 
                                         'ozone': int, 'no2': int, 'so2': int, 'co': int})
        else:
            # Location-specific data
            loc_seed = hash(location) % 1000
            np.random.seed(loc_seed)
            
            # Create realistic AQI pattern with daily cycles
            base_aqi = 50 + 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
            trend = np.linspace(0, 10, len(dates))  # Slight upward trend
            noise = np.random.normal(0, 8, len(dates))
            
            aqi_values = np.maximum(10, base_aqi + trend + noise)
            
            data = {
                'timestamp': dates,
                'location': location,
                'aqi': aqi_values.astype(int),
                'pm25': (aqi_values * 0.6).astype(int),
                'pm10': (aqi_values * 0.8).astype(int),
                'ozone': (aqi_values * 0.4).astype(int),
                'no2': (aqi_values * 0.3).astype(int),
                'so2': (aqi_values * 0.2).astype(int),
                'co': (aqi_values * 0.1).astype(int)
            }
            
            return pd.DataFrame(data)
    
    def get_disease_data(self, location: str, date_range: Tuple) -> pd.DataFrame:
        """Generate disease outbreak data"""
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                end=datetime.now(), freq='D')
        
        diseases = ['Influenza', 'COVID-19', 'Norovirus', 'RSV', 'Pneumonia']
        
        if location == "All Locations":
            data_points = []
            for loc in self.locations:
                loc_seed = hash(loc) % 1000
                np.random.seed(loc_seed)
                
                for disease in diseases:
                    # Create outbreak patterns
                    base_cases = np.random.poisson(20, len(dates))
                    # Add some outbreak spikes
                    outbreak_days = np.random.choice(len(dates), size=2, replace=False)
                    for day in outbreak_days:
                        if day < len(dates) - 3:
                            base_cases[day:day+3] *= np.random.randint(3, 8)
                    
                    for i, date in enumerate(dates):
                        data_points.append({
                            'timestamp': date,
                            'location': loc,
                            'disease': disease,
                            'cases': int(base_cases[i]),
                            'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], 
                                                       p=[0.6, 0.3, 0.1])
                        })
            
            df = pd.DataFrame(data_points)
            # Sum cases across locations and diseases
            agg_df = df.groupby('timestamp')['cases'].sum().reset_index()
            return agg_df
        else:
            data_points = []
            loc_seed = hash(location) % 1000
            np.random.seed(loc_seed)
            
            for disease in diseases:
                base_cases = np.random.poisson(15, len(dates))
                # Add outbreak pattern for this location
                outbreak_days = np.random.choice(len(dates), size=1, replace=False)
                for day in outbreak_days:
                    if day < len(dates) - 5:
                        base_cases[day:day+5] *= np.random.randint(2, 6)
                
                for i, date in enumerate(dates):
                    data_points.append({
                        'timestamp': date,
                        'location': location,
                        'disease': disease,
                        'cases': int(base_cases[i]),
                        'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], 
                                                   p=[0.6, 0.3, 0.1])
                    })
            
            df = pd.DataFrame(data_points)
            # Sum across diseases for location
            agg_df = df.groupby('timestamp')['cases'].sum().reset_index()
            return agg_df
    
    def get_hospital_capacity_data(self, location: str, date_range: Tuple) -> pd.DataFrame:
        """Generate hospital capacity data"""
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                end=datetime.now(), freq='D')
        
        if location == "All Locations":
            data_points = []
            for loc in self.locations:
                loc_seed = hash(loc) % 1000
                np.random.seed(loc_seed)
                
                # Hospital capacity varies with disease outbreaks
                base_capacity = 75 + 15 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
                noise = np.random.normal(0, 5, len(dates))
                capacity = np.clip(base_capacity + noise, 50, 100)
                
                for i, date in enumerate(dates):
                    data_points.append({
                        'timestamp': date,
                        'location': loc,
                        'bed_occupancy': int(capacity[i]),
                        'icu_occupancy': int(capacity[i] * 0.8),
                        'emergency_visits': int(100 + capacity[i] * 2),
                        'available_beds': int((100 - capacity[i]) * 3),
                        'staff_availability': int(100 - capacity[i] * 0.3)
                    })
            
            df = pd.DataFrame(data_points)
            # Average across locations
            agg_df = df.groupby('timestamp').agg({
                'bed_occupancy': 'mean',
                'icu_occupancy': 'mean',
                'emergency_visits': 'mean',
                'available_beds': 'mean',
                'staff_availability': 'mean'
            }).reset_index()
            
            return agg_df.round(0).astype({
                'bed_occupancy': int, 'icu_occupancy': int, 'emergency_visits': int,
                'available_beds': int, 'staff_availability': int
            })
        else:
            loc_seed = hash(location) % 1000
            np.random.seed(loc_seed)
            
            # Location-specific capacity data
            base_capacity = 70 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
            seasonal_trend = 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30)
            noise = np.random.normal(0, 4, len(dates))
            
            capacity = np.clip(base_capacity + seasonal_trend + noise, 45, 98)
            
            data = {
                'timestamp': dates,
                'location': location,
                'bed_occupancy': capacity.astype(int),
                'icu_occupancy': (capacity * 0.75).astype(int),
                'emergency_visits': (80 + capacity * 2.5).astype(int),
                'available_beds': ((100 - capacity) * 2.5).astype(int),
                'staff_availability': (100 - capacity * 0.4).astype(int)
            }
            
            return pd.DataFrame(data)
    
    def get_geospatial_data(self) -> pd.DataFrame:
        """Generate geospatial data for mapping"""
        # Coordinates for major US cities
        coordinates = {
            "New York, NY": [40.7128, -74.0060],
            "Los Angeles, CA": [34.0522, -118.2437],
            "Chicago, IL": [41.8781, -87.6298],
            "Houston, TX": [29.7604, -95.3698],
            "Phoenix, AZ": [33.4484, -112.0740],
            "Philadelphia, PA": [39.9526, -75.1652],
            "San Antonio, TX": [29.4241, -98.4936],
            "San Diego, CA": [32.7157, -117.1611],
            "Dallas, TX": [32.7767, -96.7970],
            "San Jose, CA": [37.3382, -121.8863]
        }
        
        data_points = []
        for location, coords in coordinates.items():
            loc_seed = hash(location) % 1000
            np.random.seed(loc_seed)
            
            # Get current metrics for each location
            metrics = self.get_current_metrics(location)
            
            data_points.append({
                'location': location,
                'latitude': coords[0],
                'longitude': coords[1],
                'aqi': metrics['aqi'],
                'disease_cases': metrics['disease_cases'],
                'hospital_capacity': metrics['hospital_capacity'],
                'risk_level': metrics['risk_level'],
                'population': np.random.randint(500000, 8000000)
            })
        
        return pd.DataFrame(data_points)
