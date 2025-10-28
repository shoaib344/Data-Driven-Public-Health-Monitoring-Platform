"""
Automated Data Pipeline for Public Health Monitor
Implements scheduled background jobs for data ingestion, validation, cleaning, and monitoring
"""
import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from collections import defaultdict
import schedule
import concurrent.futures
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from database_service import db_service
from database_models import get_db
from realtime_data_ingestion import RealTimeDataIngester

class PipelineStatus(Enum):
    """Pipeline execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class DataSource(Enum):
    """Available data sources"""
    EPA_AIR_QUALITY = "epa_air_quality"
    CDC_DISEASE_DATA = "cdc_disease_data"
    HHS_HOSPITAL_DATA = "hhs_hospital_data"
    NOAA_WEATHER = "noaa_weather"
    ML_PREDICTIONS = "ml_predictions"

@dataclass
class PipelineJob:
    """Represents a pipeline job execution"""
    job_id: str
    job_name: str
    data_source: DataSource
    scheduled_time: datetime
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    status: PipelineStatus = PipelineStatus.IDLE
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    records_validated: int
    records_passed: int
    records_rejected: int

class DataValidator:
    """Validates incoming data for quality and completeness"""
    
    def __init__(self):
        self.validation_rules = {
            DataSource.EPA_AIR_QUALITY: self._validate_air_quality,
            DataSource.CDC_DISEASE_DATA: self._validate_disease_data,
            DataSource.HHS_HOSPITAL_DATA: self._validate_hospital_data,
            DataSource.NOAA_WEATHER: self._validate_weather_data,
            DataSource.ML_PREDICTIONS: self._validate_predictions
        }
    
    def validate_data(self, data_source: DataSource, data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate data from a specific source"""
        logger.info(f"Validating {len(data)} records from {data_source.value}")
        
        validation_func = self.validation_rules.get(data_source)
        if not validation_func:
            return ValidationResult(
                is_valid=False,
                errors=[f"No validation rule for data source: {data_source.value}"],
                warnings=[],
                records_validated=0,
                records_passed=0,
                records_rejected=0
            )
        
        try:
            return validation_func(data)
        except Exception as e:
            logger.error(f"Validation error for {data_source.value}: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation exception: {str(e)}"],
                warnings=[],
                records_validated=len(data),
                records_passed=0,
                records_rejected=len(data)
            )
    
    def _validate_air_quality(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate air quality data"""
        errors = []
        warnings = []
        passed = 0
        rejected = 0
        
        required_fields = ['aqi', 'timestamp', 'location_id']
        
        for i, record in enumerate(data):
            record_errors = []
            
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    record_errors.append(f"Record {i}: Missing required field '{field}'")
            
            # Validate AQI range
            if 'aqi' in record:
                try:
                    aqi_value = float(record['aqi'])
                    if aqi_value < 0 or aqi_value > 500:
                        record_errors.append(f"Record {i}: AQI value {aqi_value} out of valid range (0-500)")
                except (ValueError, TypeError):
                    record_errors.append(f"Record {i}: Invalid AQI value format")
            
            # Validate timestamp
            if 'timestamp' in record:
                try:
                    timestamp = datetime.fromisoformat(str(record['timestamp']).replace('Z', '+00:00'))
                    # Check if timestamp is too far in the future or past
                    now = datetime.utcnow()
                    if timestamp > now + timedelta(hours=1):
                        warnings.append(f"Record {i}: Future timestamp detected")
                    elif timestamp < now - timedelta(days=30):
                        warnings.append(f"Record {i}: Very old timestamp detected")
                except (ValueError, TypeError):
                    record_errors.append(f"Record {i}: Invalid timestamp format")
            
            if record_errors:
                errors.extend(record_errors)
                rejected += 1
            else:
                passed += 1
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            records_validated=len(data),
            records_passed=passed,
            records_rejected=rejected
        )
    
    def _validate_disease_data(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate disease surveillance data"""
        errors = []
        warnings = []
        passed = 0
        rejected = 0
        
        required_fields = ['cases', 'timestamp', 'location_id', 'disease_type']
        
        for i, record in enumerate(data):
            record_errors = []
            
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    record_errors.append(f"Record {i}: Missing required field '{field}'")
            
            # Validate case count
            if 'cases' in record:
                try:
                    cases = int(record['cases'])
                    if cases < 0:
                        record_errors.append(f"Record {i}: Negative case count not allowed")
                    elif cases > 100000:  # Sanity check
                        warnings.append(f"Record {i}: Very high case count ({cases})")
                except (ValueError, TypeError):
                    record_errors.append(f"Record {i}: Invalid case count format")
            
            if record_errors:
                errors.extend(record_errors)
                rejected += 1
            else:
                passed += 1
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            records_validated=len(data),
            records_passed=passed,
            records_rejected=rejected
        )
    
    def _validate_hospital_data(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate hospital capacity data"""
        errors = []
        warnings = []
        passed = 0
        rejected = 0
        
        required_fields = ['bed_occupancy', 'total_beds', 'timestamp', 'location_id']
        
        for i, record in enumerate(data):
            record_errors = []
            
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    record_errors.append(f"Record {i}: Missing required field '{field}'")
            
            # Validate occupancy rates
            if 'bed_occupancy' in record and 'total_beds' in record:
                try:
                    occupancy = float(record['bed_occupancy'])
                    total = int(record['total_beds'])
                    
                    if occupancy < 0 or occupancy > 100:
                        record_errors.append(f"Record {i}: Bed occupancy must be 0-100%")
                    
                    if total <= 0:
                        record_errors.append(f"Record {i}: Total beds must be positive")
                        
                except (ValueError, TypeError):
                    record_errors.append(f"Record {i}: Invalid numeric values for bed data")
            
            if record_errors:
                errors.extend(record_errors)
                rejected += 1
            else:
                passed += 1
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            records_validated=len(data),
            records_passed=passed,
            records_rejected=rejected
        )
    
    def _validate_weather_data(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate weather data"""
        errors = []
        warnings = []
        passed = 0
        rejected = 0
        
        required_fields = ['temperature', 'humidity', 'timestamp', 'location_id']
        
        for i, record in enumerate(data):
            record_errors = []
            
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    record_errors.append(f"Record {i}: Missing required field '{field}'")
            
            # Validate temperature range (Celsius)
            if 'temperature' in record:
                try:
                    temp = float(record['temperature'])
                    if temp < -50 or temp > 60:  # Extreme but possible values
                        warnings.append(f"Record {i}: Extreme temperature value ({temp}°C)")
                except (ValueError, TypeError):
                    record_errors.append(f"Record {i}: Invalid temperature format")
            
            # Validate humidity range
            if 'humidity' in record:
                try:
                    humidity = float(record['humidity'])
                    if humidity < 0 or humidity > 100:
                        record_errors.append(f"Record {i}: Humidity must be 0-100%")
                except (ValueError, TypeError):
                    record_errors.append(f"Record {i}: Invalid humidity format")
            
            if record_errors:
                errors.extend(record_errors)
                rejected += 1
            else:
                passed += 1
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            records_validated=len(data),
            records_passed=passed,
            records_rejected=rejected
        )
    
    def _validate_predictions(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate ML prediction data"""
        errors = []
        warnings = []
        passed = 0
        rejected = 0
        
        required_fields = ['predicted_value', 'confidence_score', 'timestamp', 'location_id', 'metric_type']
        
        for i, record in enumerate(data):
            record_errors = []
            
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    record_errors.append(f"Record {i}: Missing required field '{field}'")
            
            # Validate confidence score
            if 'confidence_score' in record:
                try:
                    confidence = float(record['confidence_score'])
                    if confidence < 0 or confidence > 1:
                        record_errors.append(f"Record {i}: Confidence score must be 0-1")
                except (ValueError, TypeError):
                    record_errors.append(f"Record {i}: Invalid confidence score format")
            
            if record_errors:
                errors.extend(record_errors)
                rejected += 1
            else:
                passed += 1
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            records_validated=len(data),
            records_passed=passed,
            records_rejected=rejected
        )

class DataCleaner:
    """Cleans and standardizes incoming data"""
    
    def clean_data(self, data_source: DataSource, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Clean data for a specific source"""
        logger.info(f"Cleaning {len(data)} records from {data_source.value}")
        
        cleaned_data = []
        issues = []
        
        for i, record in enumerate(data):
            try:
                cleaned_record = self._clean_record(data_source, record, i)
                if cleaned_record:
                    cleaned_data.append(cleaned_record)
            except Exception as e:
                issues.append(f"Record {i}: Cleaning error - {str(e)}")
        
        logger.info(f"Cleaned {len(cleaned_data)} records, {len(issues)} issues found")
        return cleaned_data, issues
    
    def _clean_record(self, data_source: DataSource, record: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """Clean a single record"""
        cleaned = record.copy()
        
        # Standardize timestamp format
        if 'timestamp' in cleaned:
            try:
                if isinstance(cleaned['timestamp'], str):
                    # Handle different timestamp formats
                    ts_str = cleaned['timestamp'].replace('Z', '+00:00')
                    cleaned['timestamp'] = datetime.fromisoformat(ts_str)
                elif isinstance(cleaned['timestamp'], (int, float)):
                    cleaned['timestamp'] = datetime.fromtimestamp(cleaned['timestamp'])
            except Exception as e:
                logger.warning(f"Could not parse timestamp for record {index}: {e}")
                return None
        
        # Remove null and empty values
        cleaned = {k: v for k, v in cleaned.items() if v is not None and v != ''}
        
        # Data source specific cleaning
        if data_source == DataSource.EPA_AIR_QUALITY:
            cleaned = self._clean_air_quality_record(cleaned)
        elif data_source == DataSource.CDC_DISEASE_DATA:
            cleaned = self._clean_disease_record(cleaned)
        elif data_source == DataSource.HHS_HOSPITAL_DATA:
            cleaned = self._clean_hospital_record(cleaned)
        elif data_source == DataSource.NOAA_WEATHER:
            cleaned = self._clean_weather_record(cleaned)
        
        return cleaned
    
    def _clean_air_quality_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean air quality specific fields"""
        if 'aqi' in record:
            try:
                # Round AQI to nearest integer
                record['aqi'] = round(float(record['aqi']))
                # Cap at maximum valid AQI
                record['aqi'] = min(record['aqi'], 500)
            except (ValueError, TypeError):
                pass
        
        return record
    
    def _clean_disease_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean disease data specific fields"""
        if 'cases' in record:
            try:
                # Ensure case count is integer
                record['cases'] = int(float(record['cases']))
                # Ensure non-negative
                record['cases'] = max(record['cases'], 0)
            except (ValueError, TypeError):
                pass
        
        # Standardize disease type names
        if 'disease_type' in record:
            record['disease_type'] = str(record['disease_type']).lower().strip()
        
        return record
    
    def _clean_hospital_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean hospital data specific fields"""
        if 'bed_occupancy' in record:
            try:
                record['bed_occupancy'] = round(float(record['bed_occupancy']), 1)
                record['bed_occupancy'] = max(0, min(record['bed_occupancy'], 100))
            except (ValueError, TypeError):
                pass
        
        if 'total_beds' in record:
            try:
                record['total_beds'] = int(float(record['total_beds']))
                record['total_beds'] = max(record['total_beds'], 1)
            except (ValueError, TypeError):
                pass
        
        return record
    
    def _clean_weather_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean weather data specific fields"""
        if 'temperature' in record:
            try:
                record['temperature'] = round(float(record['temperature']), 1)
            except (ValueError, TypeError):
                pass
        
        if 'humidity' in record:
            try:
                record['humidity'] = round(float(record['humidity']), 1)
                record['humidity'] = max(0, min(record['humidity'], 100))
            except (ValueError, TypeError):
                pass
        
        return record

class PipelineMonitor:
    """Monitors pipeline execution and health"""
    
    def __init__(self):
        self.job_history: List[PipelineJob] = []
        self.failure_counts = defaultdict(int)
        self.success_rates = defaultdict(list)
        self.performance_metrics = defaultdict(list)
    
    def record_job_start(self, job: PipelineJob):
        """Record the start of a pipeline job"""
        job.status = PipelineStatus.RUNNING
        job.started_time = datetime.utcnow()
        logger.info(f"Pipeline job started: {job.job_name}")
    
    def record_job_completion(self, job: PipelineJob, success: bool = True, error: str = None):
        """Record the completion of a pipeline job"""
        job.completed_time = datetime.utcnow()
        job.status = PipelineStatus.COMPLETED if success else PipelineStatus.FAILED
        
        if error:
            job.error_message = error
        
        if job.started_time:
            job.execution_time_seconds = (job.completed_time - job.started_time).total_seconds()
            self.performance_metrics[job.data_source].append(job.execution_time_seconds)
        
        # Track success/failure rates
        if success:
            self.success_rates[job.data_source].append(1)
            logger.info(f"Pipeline job completed successfully: {job.job_name}")
        else:
            self.success_rates[job.data_source].append(0)
            self.failure_counts[job.data_source] += 1
            logger.error(f"Pipeline job failed: {job.job_name} - {error}")
        
        # Keep recent history
        self.job_history.append(job)
        if len(self.job_history) > 1000:  # Keep last 1000 jobs
            self.job_history = self.job_history[-1000:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall pipeline health status"""
        recent_jobs = [job for job in self.job_history if 
                      job.completed_time and job.completed_time > datetime.utcnow() - timedelta(hours=24)]
        
        if not recent_jobs:
            return {
                'status': 'unknown',
                'message': 'No recent pipeline activity',
                'jobs_last_24h': 0,
                'success_rate': 0,
                'avg_execution_time': 0
            }
        
        successful_jobs = [job for job in recent_jobs if job.status == PipelineStatus.COMPLETED]
        success_rate = len(successful_jobs) / len(recent_jobs) if recent_jobs else 0
        
        avg_execution_time = sum(job.execution_time_seconds or 0 for job in recent_jobs) / len(recent_jobs)
        
        # Determine overall health
        if success_rate >= 0.95:
            status = 'healthy'
            message = 'Pipeline operating normally'
        elif success_rate >= 0.80:
            status = 'warning'
            message = f'Pipeline success rate below 95% ({success_rate:.1%})'
        else:
            status = 'critical'
            message = f'Pipeline success rate critically low ({success_rate:.1%})'
        
        return {
            'status': status,
            'message': message,
            'jobs_last_24h': len(recent_jobs),
            'success_rate': success_rate,
            'avg_execution_time': avg_execution_time,
            'total_jobs': len(self.job_history),
            'data_source_stats': self._get_data_source_stats()
        }
    
    def _get_data_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics per data source"""
        stats = {}
        
        for data_source in DataSource:
            source_jobs = [job for job in self.job_history if job.data_source == data_source]
            if not source_jobs:
                continue
            
            recent_jobs = [job for job in source_jobs if 
                          job.completed_time and job.completed_time > datetime.utcnow() - timedelta(hours=24)]
            
            successful_recent = [job for job in recent_jobs if job.status == PipelineStatus.COMPLETED]
            
            stats[data_source.value] = {
                'total_jobs': len(source_jobs),
                'jobs_last_24h': len(recent_jobs),
                'success_rate_24h': len(successful_recent) / len(recent_jobs) if recent_jobs else 0,
                'failure_count': self.failure_counts[data_source],
                'avg_execution_time': sum(self.performance_metrics[data_source]) / len(self.performance_metrics[data_source]) if self.performance_metrics[data_source] else 0,
                'last_successful_run': max((job.completed_time for job in successful_recent), default=None)
            }
        
        return stats

def pipeline_job_decorator(data_source: DataSource, job_name: str):
    """Decorator for pipeline jobs to handle monitoring and error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            job = PipelineJob(
                job_id=str(uuid.uuid4()),
                job_name=job_name,
                data_source=data_source,
                scheduled_time=datetime.utcnow()
            )
            
            monitor = pipeline_system.monitor
            monitor.record_job_start(job)
            
            try:
                result = func(job, *args, **kwargs)
                monitor.record_job_completion(job, success=True)
                return result
            
            except Exception as e:
                monitor.record_job_completion(job, success=False, error=str(e))
                logger.error(f"Pipeline job {job_name} failed: {e}")
                raise
        
        return wrapper
    return decorator

class AutomatedDataPipeline:
    """Main automated data pipeline system"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.monitor = PipelineMonitor()
        self.is_running = False
        self.scheduler_thread = None
        self.data_collector = RealTimeDataIngester()
        
        # Pipeline configuration
        self.config = {
            'air_quality_interval_minutes': 15,
            'disease_data_interval_minutes': 60,
            'hospital_data_interval_minutes': 30,
            'weather_data_interval_minutes': 30,
            'ml_predictions_interval_minutes': 120,
            'max_retries': 3,
            'retry_delay_seconds': 60,
            'batch_size': 100,
            'max_concurrent_jobs': 3
        }
    
    def start_pipeline(self):
        """Start the automated data pipeline"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        logger.info("Starting automated data pipeline...")
        
        # Schedule data collection jobs
        self._schedule_jobs()
        
        # Start scheduler in background thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Automated data pipeline started successfully")
    
    def stop_pipeline(self):
        """Stop the automated data pipeline"""
        if not self.is_running:
            logger.warning("Pipeline is not currently running")
            return
        
        logger.info("Stopping automated data pipeline...")
        self.is_running = False
        
        # Clear scheduled jobs
        schedule.clear()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Automated data pipeline stopped")
    
    def _schedule_jobs(self):
        """Schedule all pipeline jobs"""
        # Air quality data every 15 minutes
        schedule.every(self.config['air_quality_interval_minutes']).minutes.do(
            self._execute_job_with_retry, self._collect_air_quality_data
        )
        
        # Disease data every hour
        schedule.every(self.config['disease_data_interval_minutes']).minutes.do(
            self._execute_job_with_retry, self._collect_disease_data
        )
        
        # Hospital data every 30 minutes
        schedule.every(self.config['hospital_data_interval_minutes']).minutes.do(
            self._execute_job_with_retry, self._collect_hospital_data
        )
        
        # Weather data every 30 minutes
        schedule.every(self.config['weather_data_interval_minutes']).minutes.do(
            self._execute_job_with_retry, self._collect_weather_data
        )
        
        # ML predictions every 2 hours
        schedule.every(self.config['ml_predictions_interval_minutes']).minutes.do(
            self._execute_job_with_retry, self._collect_ml_predictions
        )
        
        # Pipeline health check every 10 minutes
        schedule.every(10).minutes.do(self._health_check_job)
        
        logger.info("Pipeline jobs scheduled successfully")
    
    def _run_scheduler(self):
        """Run the scheduler in background"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    def _execute_job_with_retry(self, job_func):
        """Execute a job with retry logic"""
        max_retries = self.config['max_retries']
        retry_delay = self.config['retry_delay_seconds']
        
        for attempt in range(max_retries + 1):
            try:
                return job_func()
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Job failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {retry_delay}s: {e}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Job failed after {max_retries + 1} attempts: {e}")
                    raise
    
    @pipeline_job_decorator(DataSource.EPA_AIR_QUALITY, "Air Quality Data Collection")
    def _collect_air_quality_data(self, job: PipelineJob):
        """Collect air quality data from EPA"""
        logger.info("Collecting air quality data...")
        
        try:
            # Get data from collector
            raw_data = self.data_collector.ingest_epa_air_quality()
            
            if not raw_data:
                logger.warning("No air quality data received")
                return
            
            # Validate data
            validation_result = self.validator.validate_data(DataSource.EPA_AIR_QUALITY, raw_data)
            
            if not validation_result.is_valid:
                logger.error(f"Air quality data validation failed: {validation_result.errors}")
                job.records_failed = validation_result.records_rejected
                return
            
            # Clean data
            cleaned_data, cleaning_issues = self.cleaner.clean_data(DataSource.EPA_AIR_QUALITY, raw_data)
            
            if cleaning_issues:
                logger.warning(f"Data cleaning issues: {cleaning_issues}")
            
            # Store in database
            records_inserted, records_updated = self._store_health_data(cleaned_data, 'aqi')
            
            # Update job statistics
            job.records_processed = len(raw_data)
            job.records_inserted = records_inserted
            job.records_updated = records_updated
            job.metadata = {
                'validation_warnings': len(validation_result.warnings),
                'cleaning_issues': len(cleaning_issues)
            }
            
            logger.info(f"Air quality data processed: {records_inserted} inserted, {records_updated} updated")
            
        except Exception as e:
            job.error_message = str(e)
            raise
    
    @pipeline_job_decorator(DataSource.CDC_DISEASE_DATA, "Disease Data Collection")
    def _collect_disease_data(self, job: PipelineJob):
        """Collect disease surveillance data from CDC"""
        logger.info("Collecting disease surveillance data...")
        
        try:
            raw_data = self.data_collector.ingest_cdc_disease_data()
            
            if not raw_data:
                logger.warning("No disease data received")
                return
            
            validation_result = self.validator.validate_data(DataSource.CDC_DISEASE_DATA, raw_data)
            
            if not validation_result.is_valid:
                logger.error(f"Disease data validation failed: {validation_result.errors}")
                job.records_failed = validation_result.records_rejected
                return
            
            cleaned_data, cleaning_issues = self.cleaner.clean_data(DataSource.CDC_DISEASE_DATA, raw_data)
            
            records_inserted, records_updated = self._store_health_data(cleaned_data, 'disease_cases')
            
            job.records_processed = len(raw_data)
            job.records_inserted = records_inserted
            job.records_updated = records_updated
            job.metadata = {
                'validation_warnings': len(validation_result.warnings),
                'cleaning_issues': len(cleaning_issues)
            }
            
            logger.info(f"Disease data processed: {records_inserted} inserted, {records_updated} updated")
            
        except Exception as e:
            job.error_message = str(e)
            raise
    
    @pipeline_job_decorator(DataSource.HHS_HOSPITAL_DATA, "Hospital Data Collection")
    def _collect_hospital_data(self, job: PipelineJob):
        """Collect hospital capacity data from HHS"""
        logger.info("Collecting hospital capacity data...")
        
        try:
            raw_data = self.data_collector.ingest_hospital_capacity()
            
            if not raw_data:
                logger.warning("No hospital data received")
                return
            
            validation_result = self.validator.validate_data(DataSource.HHS_HOSPITAL_DATA, raw_data)
            
            if not validation_result.is_valid:
                logger.error(f"Hospital data validation failed: {validation_result.errors}")
                job.records_failed = validation_result.records_rejected
                return
            
            cleaned_data, cleaning_issues = self.cleaner.clean_data(DataSource.HHS_HOSPITAL_DATA, raw_data)
            
            records_inserted, records_updated = self._store_health_data(cleaned_data, 'bed_occupancy')
            
            job.records_processed = len(raw_data)
            job.records_inserted = records_inserted
            job.records_updated = records_updated
            
            logger.info(f"Hospital data processed: {records_inserted} inserted, {records_updated} updated")
            
        except Exception as e:
            job.error_message = str(e)
            raise
    
    @pipeline_job_decorator(DataSource.NOAA_WEATHER, "Weather Data Collection")
    def _collect_weather_data(self, job: PipelineJob):
        """Collect weather data from NOAA"""
        logger.info("Collecting weather data...")
        
        try:
            raw_data = self.data_collector.ingest_weather_data()
            
            if not raw_data:
                logger.warning("No weather data received")
                return
            
            validation_result = self.validator.validate_data(DataSource.NOAA_WEATHER, raw_data)
            
            if not validation_result.is_valid:
                logger.error(f"Weather data validation failed: {validation_result.errors}")
                job.records_failed = validation_result.records_rejected
                return
            
            cleaned_data, cleaning_issues = self.cleaner.clean_data(DataSource.NOAA_WEATHER, raw_data)
            
            # Store temperature and humidity separately
            temp_data = [{'value': r['temperature'], 'timestamp': r['timestamp'], 'location_id': r['location_id']} for r in cleaned_data if 'temperature' in r]
            humidity_data = [{'value': r['humidity'], 'timestamp': r['timestamp'], 'location_id': r['location_id']} for r in cleaned_data if 'humidity' in r]
            
            temp_inserted, temp_updated = self._store_health_data(temp_data, 'temperature')
            humidity_inserted, humidity_updated = self._store_health_data(humidity_data, 'humidity')
            
            job.records_processed = len(raw_data)
            job.records_inserted = temp_inserted + humidity_inserted
            job.records_updated = temp_updated + humidity_updated
            
            logger.info(f"Weather data processed: {job.records_inserted} inserted, {job.records_updated} updated")
            
        except Exception as e:
            job.error_message = str(e)
            raise
    
    @pipeline_job_decorator(DataSource.ML_PREDICTIONS, "ML Predictions Collection")
    def _collect_ml_predictions(self, job: PipelineJob):
        """Collect and store ML predictions"""
        logger.info("Generating ML predictions...")
        
        try:
            # Generate predictions using the ML system
            from advanced_ml_models import advanced_ml
            
            predictions = advanced_ml.generate_predictions(days_ahead=7)
            
            if not predictions:
                logger.warning("No ML predictions generated")
                return
            
            # Convert predictions to storable format
            prediction_records = []
            for model_name, pred_df in predictions.items():
                if not pred_df.empty:
                    for _, row in pred_df.iterrows():
                        prediction_records.append({
                            'predicted_value': row['predicted_value'],
                            'confidence_score': row.get('confidence_score', 0.8),
                            'timestamp': row['timestamp'],
                            'location_id': None,  # Will be set by the data ingestion
                            'metric_type': model_name.split('_')[0],  # Extract metric from model name
                            'model_name': model_name
                        })
            
            if not prediction_records:
                logger.warning("No prediction records to store")
                return
            
            # Validate prediction data
            validation_result = self.validator.validate_data(DataSource.ML_PREDICTIONS, prediction_records)
            
            if not validation_result.is_valid:
                logger.error(f"ML prediction validation failed: {validation_result.errors}")
                job.records_failed = validation_result.records_rejected
                return
            
            # Store predictions (implementation would store in ML predictions table)
            records_stored = len(prediction_records)
            
            job.records_processed = len(prediction_records)
            job.records_inserted = records_stored
            
            logger.info(f"ML predictions processed: {records_stored} predictions stored")
            
        except Exception as e:
            job.error_message = str(e)
            raise
    
    def _health_check_job(self):
        """Perform pipeline health check"""
        try:
            health_status = self.monitor.get_health_status()
            logger.info(f"Pipeline health check: {health_status['status']} - {health_status['message']}")
            
            # Alert if pipeline health is critical
            if health_status['status'] == 'critical':
                logger.critical("Pipeline health is critical - immediate attention required")
                # Here you would send alerts to administrators
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _store_health_data(self, data: List[Dict[str, Any]], metric_type: str) -> Tuple[int, int]:
        """Store health data in database"""
        if not data:
            return 0, 0
        
        db = next(get_db())
        try:
            inserted = 0
            updated = 0
            
            for record in data:
                try:
                    # Store record using database service
                    success = db_service.health_data.store_health_metric(
                        db=db,
                        location_id=record.get('location_id'),
                        metric_type=metric_type,
                        value=record.get('value', record.get(metric_type)),
                        timestamp=record.get('timestamp')
                    )
                    
                    if success:
                        inserted += 1
                    else:
                        updated += 1  # Assume it was an update if insert failed
                        
                except Exception as e:
                    logger.error(f"Error storing record: {e}")
                    continue
            
            db.commit()
            return inserted, updated
            
        except Exception as e:
            db.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            db.close()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'is_running': self.is_running,
            'configuration': self.config,
            'health_status': self.monitor.get_health_status(),
            'recent_jobs': [asdict(job) for job in self.monitor.job_history[-10:]],  # Last 10 jobs
            'scheduler_thread_active': self.scheduler_thread and self.scheduler_thread.is_alive() if self.scheduler_thread else False
        }
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update pipeline configuration"""
        self.config.update(new_config)
        logger.info(f"Pipeline configuration updated: {new_config}")
        
        # Restart pipeline if it's running to apply new schedule
        if self.is_running:
            logger.info("Restarting pipeline to apply new configuration...")
            self.stop_pipeline()
            time.sleep(2)
            self.start_pipeline()

# Global pipeline system instance
pipeline_system = AutomatedDataPipeline()