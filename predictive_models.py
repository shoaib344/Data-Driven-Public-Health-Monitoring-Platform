import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PredictiveModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame, target_column: str) -> tuple:
        """Prepare features for training"""
        # Create time-based features
        df = df.copy()
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['day_of_year'] = pd.to_datetime(df['timestamp']).dt.dayofyear
        
        # Create lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            if len(df) > lag:
                df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Rolling statistics
        for window in [6, 12, 24]:
            if len(df) > window:
                df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window).mean()
                df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window).std()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) == 0:
            return None, None, None, None
        
        # Define feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', target_column, 'location']]
        
        if not feature_cols:
            # Fallback to basic features
            feature_cols = ['hour', 'day_of_week', 'month']
        
        X = df[feature_cols]
        y = df[target_column]
        
        return X, y, feature_cols, df
    
    def train_air_quality_model(self, data: pd.DataFrame) -> dict:
        """Train air quality prediction model"""
        if data.empty:
            return {'error': 'No data available for training'}
        
        X, y, feature_cols, processed_df = self.prepare_features(data, 'aqi')
        
        if X is None or len(X) < 10:
            return {'error': 'Insufficient data for training (need at least 10 samples)'}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'LinearRegression': LinearRegression()
            }
            
            best_model = None
            best_score = float('-inf')
            best_model_name = None
            
            results = {}
            
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                    
                    mae = mean_absolute_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    
                    results[name] = {
                        'mae': mae,
                        'r2': r2,
                        'model': model
                    }
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = model
                        best_model_name = name
                        
                except Exception as e:
                    results[name] = {'error': str(e)}
            
            if best_model is None:
                return {'error': 'All models failed to train'}
            
            # Store the best model
            self.models['aqi'] = best_model
            self.scalers['aqi'] = scaler
            self.feature_columns = feature_cols
            
            return {
                'best_model': best_model_name,
                'best_score': best_score,
                'results': results,
                'feature_importance': self._get_feature_importance(best_model, feature_cols)
            }
            
        except Exception as e:
            return {'error': f'Training failed: {str(e)}'}
    
    def _get_feature_importance(self, model, feature_cols):
        """Get feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_cols, importance))
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            return dict(zip(feature_cols, importance))
        else:
            return {}
    
    def predict_air_quality(self, data: pd.DataFrame, hours_ahead: int = 24) -> pd.DataFrame:
        """Predict air quality for future hours"""
        if 'aqi' not in self.models:
            return pd.DataFrame()
        
        if data.empty:
            return pd.DataFrame()
        
        try:
            model = self.models['aqi']
            scaler = self.scalers['aqi']
            
            # Prepare the latest data point
            latest_data = data.copy().tail(100)  # Use last 100 points for context
            
            # Create features for the latest point
            X, _, feature_cols, _ = self.prepare_features(latest_data, 'aqi')
            
            if X is None or X.empty:
                return pd.DataFrame()
            
            # Generate future timestamps
            last_timestamp = pd.to_datetime(data['timestamp'].max())
            future_timestamps = pd.date_range(
                start=last_timestamp + timedelta(hours=1),
                periods=hours_ahead,
                freq='H'
            )
            
            predictions = []
            current_features = X.iloc[-1].copy()
            
            for i, future_time in enumerate(future_timestamps):
                # Update time-based features
                current_features['hour'] = future_time.hour
                current_features['day_of_week'] = future_time.dayofweek
                current_features['month'] = future_time.month
                current_features['day_of_year'] = future_time.dayofyear
                
                # Scale features
                features_array = current_features.values.reshape(1, -1)
                features_scaled = scaler.transform(features_array)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                predictions.append({
                    'timestamp': future_time,
                    'predicted_aqi': max(0, prediction),  # Ensure non-negative
                    'confidence': 'High' if i < 6 else 'Medium' if i < 12 else 'Low'
                })
                
                # Update lag features with prediction for next iteration
                if i < hours_ahead - 1:
                    # Shift lag features
                    for lag in [1, 2, 3, 6, 12, 24]:
                        lag_col = f'aqi_lag_{lag}'
                        if lag_col in current_features.index:
                            if lag == 1:
                                current_features[lag_col] = prediction
                            elif f'aqi_lag_{lag-1}' in current_features.index:
                                current_features[lag_col] = current_features[f'aqi_lag_{lag-1}']
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return pd.DataFrame()
    
    def train_disease_model(self, data: pd.DataFrame) -> dict:
        """Train disease outbreak prediction model"""
        if data.empty:
            return {'error': 'No data available for training'}
        
        # Prepare features for disease prediction
        df = data.copy()
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['week_of_year'] = pd.to_datetime(df['timestamp']).dt.isocalendar().week
        
        # Create lag features
        for lag in [1, 2, 3, 7, 14]:
            if len(df) > lag:
                df[f'cases_lag_{lag}'] = df['cases'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14]:
            if len(df) > window:
                df[f'cases_rolling_mean_{window}'] = df['cases'].rolling(window).mean()
        
        df = df.dropna()
        
        if len(df) < 10:
            return {'error': 'Insufficient data for training'}
        
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'cases', 'location']]
        
        try:
            X = df[feature_cols]
            y = df['cases']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Use Poisson-appropriate model
            model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            self.models['disease'] = model
            
            return {
                'mae': mae,
                'r2': r2,
                'model_type': 'GradientBoosting'
            }
            
        except Exception as e:
            return {'error': f'Disease model training failed: {str(e)}'}
    
    def get_model_performance(self) -> dict:
        """Get performance metrics for all trained models"""
        performance = {}
        
        for model_name in self.models:
            if model_name in ['aqi', 'disease']:
                # These would typically be stored during training
                performance[model_name] = {
                    'status': 'trained',
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
        return performance
