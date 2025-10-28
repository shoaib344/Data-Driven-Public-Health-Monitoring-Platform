"""
Database service layer for the Public Health Monitoring Platform
Provides high-level database operations and business logic
"""
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
import pandas as pd

from database_models import (
    get_db, User, UserSession, Location, HealthMetric, Alert, 
    UserAlertPreference, UserLocationPreference, AlertHistory,
    DataSource, DataIngestionLog, MLModel, ModelPrediction,
    WebSocketConnection, UserRole, AlertSeverity, AlertType
)

class AuthService:
    """Authentication and user management service"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = secrets.token_hex(32)
        pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{pwd_hash}"
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            salt, pwd_hash = hashed.split(":")
            return hashlib.sha256((password + salt).encode()).hexdigest() == pwd_hash
        except ValueError:
            return False
    
    @staticmethod
    def create_user(db: Session, username: str, email: str, password: str, 
                   role: UserRole = UserRole.PUBLIC_USER, **kwargs) -> Optional[User]:
        """Create new user account"""
        # Check if user already exists
        if db.query(User).filter(or_(User.username == username, User.email == email)).first():
            return None
        
        user = User(
            username=username,
            email=email,
            password_hash=AuthService.hash_password(password),
            role=role,
            **kwargs
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user login"""
        user = db.query(User).filter(
            or_(User.username == username, User.email == username)
        ).first()
        
        if user and user.is_active and AuthService.verify_password(password, user.password_hash):
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            return user
        return None
    
    @staticmethod
    def create_session(db: Session, user_id: str, ip_address: str = None, user_agent: str = None) -> UserSession:
        """Create user session"""
        session_token = secrets.token_urlsafe(64)
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            expires_at=datetime.utcnow() + timedelta(days=30),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def validate_session(db: Session, session_token: str) -> Optional[User]:
        """Validate session token and return user"""
        session = db.query(UserSession).filter(
            and_(
                UserSession.session_token == session_token,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            )
        ).first()
        
        if session:
            user = db.query(User).filter(User.id == session.user_id).first()
            if user and user.is_active:
                return user
        return None

class HealthDataService:
    """Health data management service"""
    
    @staticmethod
    def get_locations(db: Session) -> List[Location]:
        """Get all active locations"""
        return db.query(Location).filter(Location.is_active == True).all()
    
    @staticmethod
    def get_health_metrics(db: Session, location_id: str = None, metric_type: str = None,
                          start_date: datetime = None, end_date: datetime = None,
                          limit: int = 1000) -> pd.DataFrame:
        """Get health metrics as pandas DataFrame"""
        query = db.query(HealthMetric).join(Location)
        
        if location_id:
            query = query.filter(HealthMetric.location_id == location_id)
        if metric_type:
            query = query.filter(HealthMetric.metric_type == metric_type)
        if start_date:
            query = query.filter(HealthMetric.timestamp >= start_date)
        if end_date:
            query = query.filter(HealthMetric.timestamp <= end_date)
        
        metrics = query.order_by(desc(HealthMetric.timestamp)).limit(limit).all()
        
        # Convert to DataFrame
        data = []
        for metric in metrics:
            data.append({
                'id': str(metric.id),
                'location_id': str(metric.location_id),
                'location_name': metric.location.name,
                'timestamp': metric.timestamp,
                'metric_type': metric.metric_type,
                'value': metric.value,
                'unit': metric.unit,
                'source': metric.source,
                'quality_score': metric.quality_score
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def store_health_metrics(db: Session, metrics_data: List[Dict]) -> int:
        """Store multiple health metrics"""
        metrics = []
        for data in metrics_data:
            metric = HealthMetric(**data)
            metrics.append(metric)
        
        db.add_all(metrics)
        db.commit()
        return len(metrics)
    
    @staticmethod
    def get_current_metrics(db: Session, location_id: str = None) -> Dict[str, Any]:
        """Get current health metrics summary"""
        # Get latest metrics for each type
        subquery = db.query(
            HealthMetric.metric_type,
            func.max(HealthMetric.timestamp).label('latest_time')
        )
        
        if location_id:
            subquery = subquery.filter(HealthMetric.location_id == location_id)
        
        subquery = subquery.group_by(HealthMetric.metric_type).subquery()
        
        latest_metrics = db.query(HealthMetric).join(
            subquery,
            and_(
                HealthMetric.metric_type == subquery.c.metric_type,
                HealthMetric.timestamp == subquery.c.latest_time
            )
        ).all()
        
        # Organize metrics by type
        metrics = {}
        for metric in latest_metrics:
            metrics[metric.metric_type] = {
                'value': metric.value,
                'unit': metric.unit,
                'timestamp': metric.timestamp,
                'source': metric.source
            }
        
        # Calculate derived metrics
        result = {
            'aqi': metrics.get('aqi', {}).get('value', 0),
            'disease_cases': metrics.get('disease_cases', {}).get('value', 0),
            'hospital_capacity': metrics.get('bed_occupancy', {}).get('value', 0),
            'aqi_change': 0,  # Calculate from historical data if needed
            'disease_change': 0,
            'capacity_change': 0
        }
        
        # Determine risk level
        aqi = result['aqi']
        cases = result['disease_cases']
        capacity = result['hospital_capacity']
        
        risk_score = (aqi / 150 + cases / 500 + capacity / 100) / 3
        if risk_score > 0.7:
            result['risk_level'] = "HIGH"
        elif risk_score > 0.4:
            result['risk_level'] = "MEDIUM"
        else:
            result['risk_level'] = "LOW"
        
        return result

class AlertService:
    """Alert management service"""
    
    @staticmethod
    def get_active_alerts(db: Session, location_id: str = None, 
                         user_id: str = None) -> List[Alert]:
        """Get active alerts"""
        query = db.query(Alert).filter(Alert.is_active == True)
        
        if location_id:
            query = query.filter(Alert.location_id == location_id)
        
        # If user_id provided, filter by user preferences
        if user_id:
            user_prefs = db.query(UserAlertPreference).filter(
                UserAlertPreference.user_id == user_id,
                UserAlertPreference.is_active == True
            ).all()
            
            if user_prefs:
                alert_types = [pref.alert_type for pref in user_prefs]
                query = query.filter(Alert.alert_type.in_(alert_types))
        
        return query.order_by(desc(Alert.created_at)).all()
    
    @staticmethod
    def create_alert(db: Session, location_id: str, alert_type: AlertType,
                    severity: AlertSeverity, title: str, message: str,
                    threshold_value: float = None, current_value: float = None,
                    metric_unit: str = None) -> Alert:
        """Create new alert"""
        alert = Alert(
            location_id=location_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            threshold_value=threshold_value,
            current_value=current_value,
            metric_unit=metric_unit
        )
        
        db.add(alert)
        db.commit()
        db.refresh(alert)
        return alert
    
    @staticmethod
    def resolve_alert(db: Session, alert_id: str, resolved_by_user_id: str) -> bool:
        """Resolve an alert"""
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.is_active = False
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by_user_id
            db.commit()
            return True
        return False

class UserPreferenceService:
    """User preference management service"""
    
    @staticmethod
    def get_user_alert_preferences(db: Session, user_id: str) -> List[UserAlertPreference]:
        """Get user alert preferences"""
        return db.query(UserAlertPreference).filter(
            UserAlertPreference.user_id == user_id,
            UserAlertPreference.is_active == True
        ).all()
    
    @staticmethod
    def set_alert_preference(db: Session, user_id: str, alert_type: AlertType,
                           severity_threshold: AlertSeverity, email_enabled: bool = True,
                           sms_enabled: bool = False, push_enabled: bool = True) -> UserAlertPreference:
        """Set or update alert preference"""
        pref = db.query(UserAlertPreference).filter(
            UserAlertPreference.user_id == user_id,
            UserAlertPreference.alert_type == alert_type
        ).first()
        
        if pref:
            pref.severity_threshold = severity_threshold
            pref.email_enabled = email_enabled
            pref.sms_enabled = sms_enabled
            pref.push_enabled = push_enabled
        else:
            pref = UserAlertPreference(
                user_id=user_id,
                alert_type=alert_type,
                severity_threshold=severity_threshold,
                email_enabled=email_enabled,
                sms_enabled=sms_enabled,
                push_enabled=push_enabled
            )
            db.add(pref)
        
        db.commit()
        db.refresh(pref)
        return pref
    
    @staticmethod
    def get_user_locations(db: Session, user_id: str) -> List[Location]:
        """Get user's preferred locations"""
        user_locs = db.query(UserLocationPreference).filter(
            UserLocationPreference.user_id == user_id
        ).all()
        
        return [ul.location for ul in user_locs]
    
    @staticmethod
    def add_user_location(db: Session, user_id: str, location_id: str,
                         is_favorite: bool = True, custom_name: str = None) -> UserLocationPreference:
        """Add location to user preferences"""
        pref = UserLocationPreference(
            user_id=user_id,
            location_id=location_id,
            is_favorite=is_favorite,
            custom_name=custom_name
        )
        
        db.add(pref)
        db.commit()
        db.refresh(pref)
        return pref

class MLModelService:
    """Machine learning model management service"""
    
    @staticmethod
    def get_active_models(db: Session, model_type: str = None) -> List[MLModel]:
        """Get active ML models"""
        query = db.query(MLModel).filter(MLModel.is_active == True)
        
        if model_type:
            query = query.filter(MLModel.model_type == model_type)
        
        return query.order_by(desc(MLModel.created_at)).all()
    
    @staticmethod
    def get_predictions(db: Session, location_id: str = None, 
                       metric_type: str = None, hours_ahead: int = 24) -> pd.DataFrame:
        """Get model predictions"""
        query = db.query(ModelPrediction).join(MLModel).filter(
            MLModel.is_active == True,
            ModelPrediction.prediction_for >= datetime.utcnow(),
            ModelPrediction.prediction_for <= datetime.utcnow() + timedelta(hours=hours_ahead)
        )
        
        if location_id:
            query = query.filter(ModelPrediction.location_id == location_id)
        if metric_type:
            query = query.filter(ModelPrediction.metric_type == metric_type)
        
        predictions = query.order_by(ModelPrediction.prediction_for).all()
        
        # Convert to DataFrame
        data = []
        for pred in predictions:
            data.append({
                'timestamp': pred.prediction_for,
                'metric_type': pred.metric_type,
                'predicted_value': pred.predicted_value,
                'confidence_score': pred.confidence_score,
                'confidence': 'High' if pred.confidence_score > 0.8 else 'Medium' if pred.confidence_score > 0.6 else 'Low'
            })
        
        return pd.DataFrame(data)

class DatabaseService:
    """Main database service aggregating all services"""
    
    def __init__(self):
        self.auth = AuthService()
        self.health_data = HealthDataService()
        self.alerts = AlertService()
        self.preferences = UserPreferenceService()
        self.ml_models = MLModelService()
    
    def get_session(self) -> Session:
        """Get database session"""
        return next(get_db())

# Global database service instance
db_service = DatabaseService()