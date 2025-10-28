"""
Database models for the Public Health Monitoring Platform
Comprehensive schema supporting users, real-time data, alerts, and analytics
"""
import os
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Enum, Index
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

# Database connection with validation
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is not set. "
        "Please ensure PostgreSQL database is configured. "
        "Check your environment variables."
    )

try:
    engine = create_engine(DATABASE_URL)
except Exception as e:
    raise RuntimeError(f"Failed to create database engine: {e}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enums
class UserRole(PyEnum):
    PUBLIC_USER = "public_user"
    HEALTH_AUTHORITY = "health_authority"
    ADMIN = "admin"

class AlertSeverity(PyEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class AlertType(PyEnum):
    AIR_QUALITY = "Air Quality"
    DISEASE_OUTBREAK = "Disease Outbreak"
    HOSPITAL_CAPACITY = "Hospital Capacity"
    TREND_ALERT = "Trend Alert"

class DataSourceType(PyEnum):
    EPA_AIR_QUALITY = "epa_air_quality"
    CDC_DISEASE = "cdc_disease"
    HOSPITAL_CAPACITY = "hospital_capacity"
    WEATHER = "weather"

# User Management
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.PUBLIC_USER)
    first_name = Column(String(50))
    last_name = Column(String(50))
    organization = Column(String(100))
    phone = Column(String(20))
    is_active = Column(Boolean, default=True)
    email_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    alert_preferences = relationship("UserAlertPreference", back_populates="user", cascade="all, delete-orphan")
    location_preferences = relationship("UserLocationPreference", back_populates="user", cascade="all, delete-orphan")
    alert_history = relationship("AlertHistory", back_populates="user", cascade="all, delete-orphan")

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    ip_address = Column(String(45))
    user_agent = Column(Text)

# Location Management
class Location(Base):
    __tablename__ = "locations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    state = Column(String(50))
    country = Column(String(50), default="USA")
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    population = Column(Integer)
    timezone = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    health_metrics = relationship("HealthMetric", back_populates="location")
    alerts = relationship("Alert", back_populates="location")
    user_preferences = relationship("UserLocationPreference", back_populates="location")
    
    __table_args__ = (
        Index('idx_location_coordinates', 'latitude', 'longitude'),
    )

# Health Data Storage
class HealthMetric(Base):
    __tablename__ = "health_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    location_id = Column(UUID(as_uuid=True), ForeignKey("locations.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    metric_type = Column(String(50), nullable=False)  # aqi, pm25, disease_cases, etc.
    value = Column(Float, nullable=False)
    unit = Column(String(20))
    source = Column(String(100))
    quality_score = Column(Float)  # Data quality indicator
    extra_data = Column(JSONB)  # Additional metric-specific data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    location = relationship("Location", back_populates="health_metrics")
    
    __table_args__ = (
        Index('idx_health_metrics_location_time', 'location_id', 'timestamp'),
        Index('idx_health_metrics_type_time', 'metric_type', 'timestamp'),
    )

# Alert System
class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    location_id = Column(UUID(as_uuid=True), ForeignKey("locations.id"), nullable=False)
    alert_type = Column(Enum(AlertType), nullable=False)
    severity = Column(Enum(AlertSeverity), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    threshold_value = Column(Float)
    current_value = Column(Float)
    metric_unit = Column(String(20))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    extra_data = Column(JSONB)
    
    # Relationships
    location = relationship("Location", back_populates="alerts")
    resolved_by_user = relationship("User", foreign_keys=[resolved_by])
    alert_history = relationship("AlertHistory", back_populates="alert")
    
    __table_args__ = (
        Index('idx_alerts_location_severity', 'location_id', 'severity'),
        Index('idx_alerts_type_created', 'alert_type', 'created_at'),
    )

# User Preferences
class UserAlertPreference(Base):
    __tablename__ = "user_alert_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    alert_type = Column(Enum(AlertType), nullable=False)
    severity_threshold = Column(Enum(AlertSeverity), nullable=False)
    email_enabled = Column(Boolean, default=True)
    sms_enabled = Column(Boolean, default=False)
    push_enabled = Column(Boolean, default=True)
    frequency_limit = Column(Integer, default=24)  # Max alerts per 24 hours
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="alert_preferences")

class UserLocationPreference(Base):
    __tablename__ = "user_location_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    location_id = Column(UUID(as_uuid=True), ForeignKey("locations.id"), nullable=False)
    is_favorite = Column(Boolean, default=True)
    custom_name = Column(String(100))
    notification_radius = Column(Float, default=10.0)  # km radius for alerts
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="location_preferences")
    location = relationship("Location", back_populates="user_preferences")

# Alert History & Tracking
class AlertHistory(Base):
    __tablename__ = "alert_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id = Column(UUID(as_uuid=True), ForeignKey("alerts.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String(50), nullable=False)  # sent, viewed, acknowledged, dismissed
    timestamp = Column(DateTime, default=datetime.utcnow)
    delivery_method = Column(String(20))  # email, sms, push, web
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    extra_data = Column(JSONB)
    
    # Relationships
    alert = relationship("Alert", back_populates="alert_history")
    user = relationship("User", back_populates="alert_history")
    
    __table_args__ = (
        Index('idx_alert_history_user_time', 'user_id', 'timestamp'),
    )

# Data Sources Configuration
class DataSource(Base):
    __tablename__ = "data_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    source_type = Column(Enum(DataSourceType), nullable=False)
    api_endpoint = Column(String(500))
    api_key_required = Column(Boolean, default=False)
    update_frequency = Column(Integer, default=3600)  # seconds
    is_active = Column(Boolean, default=True)
    last_successful_update = Column(DateTime)
    last_error = Column(Text)
    configuration = Column(JSONB)  # Source-specific config
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    data_ingestion_logs = relationship("DataIngestionLog", back_populates="data_source")

class DataIngestionLog(Base):
    __tablename__ = "data_ingestion_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    data_source_id = Column(UUID(as_uuid=True), ForeignKey("data_sources.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(20), nullable=False)  # running, success, failed
    records_processed = Column(Integer, default=0)
    records_inserted = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    error_message = Column(Text)
    execution_time = Column(Float)  # seconds
    extra_data = Column(JSONB)
    
    # Relationships
    data_source = relationship("DataSource", back_populates="data_ingestion_logs")
    
    __table_args__ = (
        Index('idx_ingestion_logs_source_time', 'data_source_id', 'started_at'),
    )

# ML Model Management
class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # air_quality, disease, capacity
    algorithm = Column(String(50), nullable=False)  # random_forest, lstm, etc.
    version = Column(String(20), nullable=False)
    accuracy_score = Column(Float)
    training_data_start = Column(DateTime)
    training_data_end = Column(DateTime)
    feature_columns = Column(JSONB)
    hyperparameters = Column(JSONB)
    is_active = Column(Boolean, default=False)
    model_path = Column(String(500))  # File storage path
    created_at = Column(DateTime, default=datetime.utcnow)
    last_retrained = Column(DateTime)
    
    # Relationships
    predictions = relationship("ModelPrediction", back_populates="model")

class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("ml_models.id"), nullable=False)
    location_id = Column(UUID(as_uuid=True), ForeignKey("locations.id"), nullable=False)
    prediction_for = Column(DateTime, nullable=False)  # Future timestamp
    metric_type = Column(String(50), nullable=False)
    predicted_value = Column(Float, nullable=False)
    confidence_score = Column(Float)
    prediction_range_low = Column(Float)
    prediction_range_high = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    actual_value = Column(Float)  # Filled in later for validation
    
    # Relationships
    model = relationship("MLModel", back_populates="predictions")
    location = relationship("Location", foreign_keys=[location_id])
    
    __table_args__ = (
        Index('idx_predictions_location_time', 'location_id', 'prediction_for'),
        Index('idx_predictions_model_type', 'model_id', 'metric_type'),
    )

# WebSocket Connection Management
class WebSocketConnection(Base):
    __tablename__ = "websocket_connections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    connection_id = Column(String(100), unique=True, nullable=False)
    connected_at = Column(DateTime, default=datetime.utcnow)
    last_ping = Column(DateTime, default=datetime.utcnow)
    subscribed_locations = Column(JSONB)  # List of location IDs
    subscribed_alert_types = Column(JSONB)  # List of alert types
    is_active = Column(Boolean, default=True)
    user_agent = Column(Text)
    ip_address = Column(String(45))

# Database utility functions
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

def seed_initial_data():
    """Seed database with initial locations and default data"""
    db = SessionLocal()
    try:
        # Check if locations already exist
        if db.query(Location).first():
            print("Database already seeded with initial data")
            return
        
        # Add initial locations
        locations = [
            {"name": "New York, NY", "latitude": 40.7128, "longitude": -74.0060, "population": 8419000, "state": "New York"},
            {"name": "Los Angeles, CA", "latitude": 34.0522, "longitude": -118.2437, "population": 3980000, "state": "California"},
            {"name": "Chicago, IL", "latitude": 41.8781, "longitude": -87.6298, "population": 2716000, "state": "Illinois"},
            {"name": "Houston, TX", "latitude": 29.7604, "longitude": -95.3698, "population": 2320000, "state": "Texas"},
            {"name": "Phoenix, AZ", "latitude": 33.4484, "longitude": -112.0740, "population": 1680000, "state": "Arizona"},
            {"name": "Philadelphia, PA", "latitude": 39.9526, "longitude": -75.1652, "population": 1580000, "state": "Pennsylvania"},
            {"name": "San Antonio, TX", "latitude": 29.4241, "longitude": -98.4936, "population": 1530000, "state": "Texas"},
            {"name": "San Diego, CA", "latitude": 32.7157, "longitude": -117.1611, "population": 1420000, "state": "California"},
            {"name": "Dallas, TX", "latitude": 32.7767, "longitude": -96.7970, "population": 1340000, "state": "Texas"},
            {"name": "San Jose, CA", "latitude": 37.3382, "longitude": -121.8863, "population": 1030000, "state": "California"}
        ]
        
        for loc_data in locations:
            location = Location(**loc_data)
            db.add(location)
        
        # Add default data sources
        data_sources = [
            {
                "name": "EPA AirNow API", 
                "source_type": DataSourceType.EPA_AIR_QUALITY,
                "api_endpoint": "https://www.airnowapi.org/aq/",
                "api_key_required": True,
                "update_frequency": 3600
            },
            {
                "name": "CDC WONDER API", 
                "source_type": DataSourceType.CDC_DISEASE,
                "api_endpoint": "https://wonder.cdc.gov/controller/",
                "update_frequency": 86400
            },
            {
                "name": "Hospital Capacity API", 
                "source_type": DataSourceType.HOSPITAL_CAPACITY,
                "api_endpoint": "https://healthdata.gov/api/",
                "update_frequency": 7200
            }
        ]
        
        for source_data in data_sources:
            data_source = DataSource(**source_data)
            db.add(data_source)
        
        db.commit()
        print("Initial data seeded successfully!")
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding initial data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    seed_initial_data()
    print("Database setup complete!")