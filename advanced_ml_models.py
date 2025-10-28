"""
Advanced Machine Learning Models for Public Health Monitoring
Implements ensemble methods, LSTM networks, and uncertainty quantification
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import joblib
import uuid
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from database_service import db_service
from database_models import get_db, MLModel, ModelPrediction

class FeatureEngineer:
    """Advanced feature engineering for health data prediction"""
    
    def __init__(self):
        self.scalers = {}
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_cols: List[str], lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lagged features for time series prediction"""
        df = df.copy()
        
        for col in target_cols:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_cols: List[str], windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """Create rolling statistics features"""
        df = df.copy()
        
        for col in target_cols:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Create interaction features between important variables"""
        df = df.copy()
        
        # Environmental interactions
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        if 'wind_speed' in df.columns and 'aqi' in df.columns:
            df['wind_aqi_interaction'] = df['wind_speed'] * df['aqi']
        
        # Health system interactions
        if 'disease_cases' in df.columns and 'bed_occupancy' in df.columns:
            df['cases_capacity_interaction'] = df['disease_cases'] * df['bed_occupancy']
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_cols: List[str], fit_scalers: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        df = self.create_time_features(df)
        df = self.create_lag_features(df, target_cols)
        df = self.create_rolling_features(df, target_cols)
        df = self.create_interaction_features(df, target_cols)
        
        # Remove non-numeric columns except timestamp
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'timestamp' in df.columns:
            numeric_cols.append('timestamp')
        
        df_numeric = df[numeric_cols]
        
        # Scale features if required
        if fit_scalers:
            feature_cols = [col for col in numeric_cols if col not in target_cols + ['timestamp']]
            if feature_cols:
                scaler = StandardScaler()
                df_numeric[feature_cols] = scaler.fit_transform(df_numeric[feature_cols])
                self.scalers['feature_scaler'] = scaler
        
        return df_numeric.fillna(method='ffill').fillna(0)

class EnsemblePredictor:
    """Ensemble model for health metrics prediction"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression(),
            'ridge': Ridge(alpha=1.0)
        }
        self.model_weights = {}
        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False
    
    def train(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Dict[str, float]:
        """Train ensemble model with cross-validation"""
        logger.info(f"Training ensemble model for {target_col}")
        
        # Feature engineering
        df_processed = self.feature_engineer.prepare_features(df, [target_col])
        
        # Remove rows with missing target values
        df_processed = df_processed.dropna(subset=[target_col])
        
        if len(df_processed) < 50:
            logger.warning(f"Insufficient data for training {target_col}: {len(df_processed)} rows")
            return {}
        
        # Prepare features and target
        feature_cols = [col for col in df_processed.columns if col not in [target_col, 'timestamp']]
        X = df_processed[feature_cols]
        y = df_processed[target_col]
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train and validate each model
        model_scores = {}
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
                model_scores[name] = -scores.mean()
                
                # Fit the model on full data
                model.fit(X, y)
                logger.info(f"Model {name} MAE: {model_scores[name]:.2f}")
                
            except Exception as e:
                logger.error(f"Error training model {name}: {e}")
                model_scores[name] = float('inf')
        
        # Calculate ensemble weights based on inverse of MAE
        total_inv_error = sum(1/score if score > 0 else 0 for score in model_scores.values() if score != float('inf'))
        
        if total_inv_error > 0:
            self.model_weights = {
                name: (1/score) / total_inv_error if score != float('inf') and score > 0 else 0
                for name, score in model_scores.items()
            }
        else:
            # Equal weights if all models failed
            self.model_weights = {name: 1/len(self.models) for name in self.models.keys()}
        
        self.is_fitted = True
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        logger.info(f"Ensemble weights: {self.model_weights}")
        return model_scores
    
    def predict_with_uncertainty(self, df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """Make predictions with uncertainty estimates"""
        if not self.is_fitted:
            logger.error("Model must be trained before prediction")
            return pd.DataFrame()
        
        try:
            # Feature engineering
            df_processed = self.feature_engineer.prepare_features(df, [self.target_col], fit_scalers=False)
            
            # Ensure all required features are present
            missing_features = set(self.feature_cols) - set(df_processed.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    df_processed[feature] = 0
            
            X = df_processed[self.feature_cols].fillna(0)
            
            # Generate predictions from each model
            predictions = []
            for name, model in self.models.items():
                if self.model_weights.get(name, 0) > 0:
                    pred = model.predict(X)
                    predictions.append(pred * self.model_weights[name])
            
            if not predictions:
                logger.error("No valid models for prediction")
                return pd.DataFrame()
            
            # Ensemble prediction
            ensemble_pred = np.sum(predictions, axis=0)
            
            # Calculate prediction uncertainty
            pred_std = np.std([pred/self.model_weights[name] for name, pred in zip(self.models.keys(), predictions) 
                             if self.model_weights.get(name, 0) > 0], axis=0)
            
            # Create future timestamps
            last_timestamp = df_processed['timestamp'].max()
            future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, days_ahead * 24 + 1)]
            
            # Create results DataFrame
            results = pd.DataFrame({
                'timestamp': future_timestamps[:len(ensemble_pred)],
                'predicted_value': ensemble_pred,
                'uncertainty': pred_std,
                'confidence_lower': ensemble_pred - 1.96 * pred_std,
                'confidence_upper': ensemble_pred + 1.96 * pred_std,
                'confidence_score': np.maximum(0, np.minimum(1, 1 - (pred_std / np.abs(ensemble_pred))))
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return pd.DataFrame()

class LSTMPredictor:
    """LSTM Neural Network for time series prediction"""
    
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
        # Check if TensorFlow/Keras is available
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            self.tf = tf
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.Dense = Dense
            self.Dropout = Dropout
            self.has_tensorflow = True
        except ImportError:
            logger.warning("TensorFlow not available. LSTM predictions will use ensemble fallback.")
            self.has_tensorflow = False
    
    def prepare_lstm_data(self, data: np.array) -> Tuple[np.array, np.array]:
        """Prepare data for LSTM training"""
        if not self.has_tensorflow:
            return np.array([]), np.array([])
        
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """Train LSTM model"""
        if not self.has_tensorflow:
            logger.warning("TensorFlow not available for LSTM training")
            return {'lstm_error': 'tensorflow_not_available'}
        
        logger.info(f"Training LSTM model for {target_col}")
        
        try:
            # Prepare data
            data = df[target_col].values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(data)
            
            X, y = self.prepare_lstm_data(scaled_data.flatten())
            
            if len(X) < 50:
                logger.warning(f"Insufficient data for LSTM training: {len(X)} sequences")
                return {'lstm_error': 'insufficient_data'}
            
            # Build LSTM model
            self.model = self.Sequential([
                self.LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
                self.Dropout(0.2),
                self.LSTM(50, return_sequences=False),
                self.Dropout(0.2),
                self.Dense(25),
                self.Dense(1)
            ])
            
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model
            X = X.reshape((X.shape[0], X.shape[1], 1))
            history = self.model.fit(X, y, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
            
            self.is_fitted = True
            self.target_col = target_col
            
            # Return training metrics
            final_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else final_loss
            
            return {
                'training_loss': final_loss,
                'validation_loss': val_loss
            }
            
        except Exception as e:
            logger.error(f"LSTM training error: {e}")
            return {'lstm_error': str(e)}
    
    def predict(self, df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """Make LSTM predictions"""
        if not self.has_tensorflow or not self.is_fitted:
            logger.warning("LSTM not available for prediction")
            return pd.DataFrame()
        
        try:
            # Get last sequence for prediction
            data = df[self.target_col].tail(self.sequence_length).values.reshape(-1, 1)
            scaled_data = self.scaler.transform(data)
            
            # Generate predictions
            predictions = []
            current_sequence = scaled_data.flatten()
            
            for _ in range(days_ahead * 24):  # Hourly predictions
                # Prepare input
                X = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, 1)
                
                # Predict next value
                pred = self.model.predict(X, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Update sequence
                current_sequence = np.append(current_sequence, pred)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions_scaled = self.scaler.inverse_transform(predictions).flatten()
            
            # Create future timestamps
            last_timestamp = df['timestamp'].max()
            future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, len(predictions_scaled) + 1)]
            
            # Create results DataFrame
            results = pd.DataFrame({
                'timestamp': future_timestamps,
                'predicted_value': predictions_scaled,
                'model_type': 'LSTM'
            })
            
            return results
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return pd.DataFrame()

class AdvancedMLSystem:
    """Main ML system orchestrating all models"""
    
    def __init__(self):
        self.ensemble_models = {}
        self.lstm_models = {}
        self.model_metadata = {}
    
    def train_all_models(self, location_id: str = None) -> Dict[str, Any]:
        """Train models for all health metrics"""
        logger.info("Starting comprehensive model training...")
        
        db = next(get_db())
        try:
            # Get training data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)  # Use last 30 days for training
            
            # Define target metrics to predict
            target_metrics = ['aqi', 'disease_cases', 'bed_occupancy', 'temperature']
            
            training_results = {}
            
            for metric in target_metrics:
                logger.info(f"Training models for {metric}")
                
                # Get data for this metric
                df = db_service.health_data.get_health_metrics(
                    db, location_id=location_id, metric_type=metric,
                    start_date=start_date, end_date=end_date, limit=2000
                )
                
                if df.empty or len(df) < 50:
                    logger.warning(f"Insufficient data for {metric}: {len(df)} rows")
                    continue
                
                # Sort by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Train ensemble model
                ensemble = EnsemblePredictor()
                ensemble_scores = ensemble.train(df, 'value')
                
                if ensemble_scores:
                    self.ensemble_models[metric] = ensemble
                    training_results[f'{metric}_ensemble'] = ensemble_scores
                    
                    # Store model metadata in database
                    self._store_model_metadata(db, metric, 'ensemble', ensemble_scores)
                
                # Train LSTM model
                if len(df) >= 100:  # More data needed for LSTM
                    lstm = LSTMPredictor()
                    lstm_scores = lstm.train(df, 'value')
                    
                    if lstm_scores and 'lstm_error' not in lstm_scores:
                        self.lstm_models[metric] = lstm
                        training_results[f'{metric}_lstm'] = lstm_scores
                        
                        # Store LSTM model metadata
                        self._store_model_metadata(db, metric, 'lstm', lstm_scores)
            
            logger.info(f"Model training complete. Trained models: {len(self.ensemble_models) + len(self.lstm_models)}")
            return training_results
            
        finally:
            db.close()
    
    def generate_predictions(self, location_id: str = None, days_ahead: int = 7) -> Dict[str, pd.DataFrame]:
        """Generate predictions for all trained models"""
        logger.info(f"Generating predictions for {days_ahead} days ahead")
        
        db = next(get_db())
        try:
            predictions = {}
            
            for metric in self.ensemble_models.keys():
                # Get recent data for prediction
                df = db_service.health_data.get_health_metrics(
                    db, location_id=location_id, metric_type=metric,
                    start_date=datetime.utcnow() - timedelta(days=7),
                    end_date=datetime.utcnow(), limit=500
                )
                
                if df.empty:
                    continue
                
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Ensemble predictions
                if metric in self.ensemble_models:
                    ensemble_pred = self.ensemble_models[metric].predict_with_uncertainty(df, days_ahead)
                    if not ensemble_pred.empty:
                        predictions[f'{metric}_ensemble'] = ensemble_pred
                        
                        # Store predictions in database
                        self._store_predictions(db, location_id, metric, ensemble_pred, 'ensemble')
                
                # LSTM predictions
                if metric in self.lstm_models:
                    lstm_pred = self.lstm_models[metric].predict(df, days_ahead)
                    if not lstm_pred.empty:
                        predictions[f'{metric}_lstm'] = lstm_pred
                        
                        # Store LSTM predictions
                        self._store_predictions(db, location_id, metric, lstm_pred, 'lstm')
            
            return predictions
            
        finally:
            db.close()
    
    def _store_model_metadata(self, db, metric: str, model_type: str, scores: Dict):
        """Store model metadata in database"""
        try:
            # Create or update ML model record
            existing_model = db.query(MLModel).filter(
                MLModel.model_name == f"{metric}_{model_type}",
                MLModel.model_type == model_type
            ).first()
            
            if existing_model:
                existing_model.accuracy_score = scores.get('rf', 0) if model_type == 'ensemble' else scores.get('training_loss', 0)
                existing_model.updated_at = datetime.utcnow()
            else:
                model_record = MLModel(
                    model_name=f"{metric}_{model_type}",
                    model_type=model_type,
                    target_metric=metric,
                    accuracy_score=scores.get('rf', 0) if model_type == 'ensemble' else scores.get('training_loss', 0),
                    hyperparameters={'scores': scores}
                )
                db.add(model_record)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error storing model metadata: {e}")
            db.rollback()
    
    def _store_predictions(self, db, location_id: str, metric: str, predictions: pd.DataFrame, model_type: str):
        """Store predictions in database"""
        try:
            # Get model ID
            model = db.query(MLModel).filter(
                MLModel.model_name == f"{metric}_{model_type}"
            ).first()
            
            if not model:
                return
            
            # Store each prediction
            for _, row in predictions.iterrows():
                prediction_record = ModelPrediction(
                    model_id=model.id,
                    location_id=location_id,
                    metric_type=metric,
                    prediction_for=row['timestamp'],
                    predicted_value=row['predicted_value'],
                    confidence_score=row.get('confidence_score', 0.8)
                )
                db.add(prediction_record)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error storing predictions: {e}")
            db.rollback()
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all trained models"""
        performance = {
            'ensemble_models': len(self.ensemble_models),
            'lstm_models': len(self.lstm_models),
            'total_models': len(self.ensemble_models) + len(self.lstm_models),
            'metrics_covered': list(set(list(self.ensemble_models.keys()) + list(self.lstm_models.keys()))),
            'model_details': {}
        }
        
        # Add ensemble model details
        for metric, model in self.ensemble_models.items():
            performance['model_details'][f'{metric}_ensemble'] = {
                'weights': model.model_weights,
                'is_fitted': model.is_fitted
            }
        
        # Add LSTM model details
        for metric, model in self.lstm_models.items():
            performance['model_details'][f'{metric}_lstm'] = {
                'sequence_length': model.sequence_length,
                'is_fitted': model.is_fitted,
                'has_tensorflow': model.has_tensorflow
            }
        
        return performance

# Global ML system instance
advanced_ml = AdvancedMLSystem()