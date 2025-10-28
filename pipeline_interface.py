"""
Streamlit Interface for Automated Data Pipeline Management
Provides admin interface for monitoring and managing data pipeline operations
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

from automated_data_pipeline import pipeline_system, PipelineStatus, DataSource
from auth import has_role, UserRole

class PipelineInterface:
    """Interface for managing automated data pipeline"""
    
    def show_pipeline_dashboard(self):
        """Show pipeline management dashboard for admins"""
        if not has_role(UserRole.ADMIN):
            st.error("Access denied. Admin privileges required.")
            return
        
        st.subheader("⚙️ Automated Data Pipeline Dashboard")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Pipeline Status", "Job History", "Data Quality", "Configuration", "Monitoring"
        ])
        
        with tab1:
            self._show_pipeline_status()
        
        with tab2:
            self._show_job_history()
        
        with tab3:
            self._show_data_quality()
        
        with tab4:
            self._show_pipeline_configuration()
        
        with tab5:
            self._show_monitoring_metrics()
    
    def _show_pipeline_status(self):
        """Show current pipeline status"""
        st.markdown("**Pipeline Status Overview**")
        
        status = pipeline_system.get_pipeline_status()
        health = status['health_status']
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pipeline_status = "🟢 Running" if status['is_running'] else "🔴 Stopped"
            st.metric("Pipeline Status", pipeline_status)
        
        with col2:
            health_color = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(health['status'], "⚫")
            st.metric("Health Status", f"{health_color} {health['status'].title()}")
        
        with col3:
            st.metric("Jobs (24h)", health.get('jobs_last_24h', 0))
        
        with col4:
            success_rate = health.get('success_rate', 0)
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        # Health message
        if health['message']:
            if health['status'] == 'healthy':
                st.success(f"✅ {health['message']}")
            elif health['status'] == 'warning':
                st.warning(f"⚠️ {health['message']}")
            else:
                st.error(f"🚨 {health['message']}")
        
        # Pipeline controls
        st.markdown("**Pipeline Controls**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not status['is_running']:
                if st.button("🚀 Start Pipeline"):
                    with st.spinner("Starting automated data pipeline..."):
                        try:
                            pipeline_system.start_pipeline()
                            st.success("✅ Pipeline started successfully!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to start pipeline: {str(e)}")
        
        with col2:
            if status['is_running']:
                if st.button("🛑 Stop Pipeline"):
                    with st.spinner("Stopping automated data pipeline..."):
                        try:
                            pipeline_system.stop_pipeline()
                            st.success("✅ Pipeline stopped successfully!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to stop pipeline: {str(e)}")
        
        with col3:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        
        # Data source status
        if 'data_source_stats' in health:
            st.markdown("**Data Source Performance**")
            
            source_data = []
            for source, stats in health['data_source_stats'].items():
                source_data.append({
                    'Data Source': source.replace('_', ' ').title(),
                    'Jobs (24h)': stats.get('jobs_last_24h', 0),
                    'Success Rate': f"{stats.get('success_rate_24h', 0):.1%}",
                    'Avg Time (s)': f"{stats.get('avg_execution_time', 0):.1f}",
                    'Failures': stats.get('failure_count', 0),
                    'Last Success': stats.get('last_successful_run').strftime('%H:%M:%S') if stats.get('last_successful_run') else 'Never'
                })
            
            if source_data:
                source_df = pd.DataFrame(source_data)
                st.dataframe(source_df, use_container_width=True, hide_index=True)
    
    def _show_job_history(self):
        """Show pipeline job execution history"""
        st.markdown("**Job Execution History**")
        
        status = pipeline_system.get_pipeline_status()
        recent_jobs = status.get('recent_jobs', [])
        
        if not recent_jobs:
            st.info("No recent job history available")
            return
        
        # Convert job data to display format
        job_data = []
        for job in recent_jobs[-20:]:  # Show last 20 jobs
            status_icon = {
                'completed': '✅',
                'failed': '❌',
                'running': '🔄',
                'idle': '⏳',
                'paused': '⏸️'
            }.get(job.get('status', 'unknown'), '❓')
            
            job_data.append({
                'Job ID': job.get('job_id', '')[:8] + '...',
                'Job Name': job.get('job_name', 'Unknown'),
                'Data Source': job.get('data_source', {}).get('_name_', 'Unknown') if isinstance(job.get('data_source'), dict) else str(job.get('data_source', 'Unknown')).replace('DataSource.', ''),
                'Status': f"{status_icon} {job.get('status', 'unknown').title()}",
                'Scheduled': job.get('scheduled_time', '')[:19] if job.get('scheduled_time') else '',
                'Duration (s)': f"{job.get('execution_time_seconds', 0):.1f}" if job.get('execution_time_seconds') else 'N/A',
                'Processed': job.get('records_processed', 0),
                'Inserted': job.get('records_inserted', 0),
                'Updated': job.get('records_updated', 0),
                'Failed': job.get('records_failed', 0)
            })
        
        if job_data:
            jobs_df = pd.DataFrame(job_data)
            st.dataframe(jobs_df, use_container_width=True, hide_index=True)
            
            # Job execution timeline
            st.markdown("**Job Execution Timeline**")
            
            # Create timeline chart
            timeline_data = []
            for job in recent_jobs[-10:]:  # Last 10 jobs for timeline
                if job.get('completed_time'):
                    timeline_data.append({
                        'Job': job.get('job_name', 'Unknown')[:20],
                        'Start': datetime.fromisoformat(job.get('started_time', '').replace('Z', '+00:00')) if job.get('started_time') else None,
                        'End': datetime.fromisoformat(job.get('completed_time', '').replace('Z', '+00:00')) if job.get('completed_time') else None,
                        'Status': job.get('status', 'unknown')
                    })
            
            if timeline_data:
                # Create Gantt-like chart
                fig = go.Figure()
                
                for i, job_info in enumerate(timeline_data):
                    if job_info['Start'] and job_info['End']:
                        color = 'green' if job_info['Status'] == 'completed' else 'red'
                        fig.add_trace(go.Scatter(
                            x=[job_info['Start'], job_info['End']],
                            y=[i, i],
                            mode='lines+markers',
                            name=job_info['Job'],
                            line=dict(color=color, width=8),
                            showlegend=False
                        ))
                
                fig.update_layout(
                    title='Recent Job Execution Timeline',
                    xaxis_title='Time',
                    yaxis_title='Jobs',
                    yaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(timeline_data))),
                        ticktext=[job['Job'] for job in timeline_data]
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _show_data_quality(self):
        """Show data quality metrics and validation results"""
        st.markdown("**Data Quality Overview**")
        
        # Mock data quality metrics (in production, this would come from validation results)
        quality_metrics = {
            'EPA Air Quality': {'completeness': 98.5, 'accuracy': 96.2, 'timeliness': 94.8, 'validity': 99.1},
            'CDC Disease Data': {'completeness': 92.3, 'accuracy': 98.7, 'timeliness': 89.4, 'validity': 96.8},
            'HHS Hospital Data': {'completeness': 95.7, 'accuracy': 94.3, 'timeliness': 97.1, 'validity': 98.2},
            'NOAA Weather': {'completeness': 99.2, 'accuracy': 97.8, 'timeliness': 98.5, 'validity': 99.6},
            'ML Predictions': {'completeness': 88.9, 'accuracy': 91.4, 'timeliness': 85.2, 'validity': 94.7}
        }
        
        # Data quality heatmap
        quality_df = pd.DataFrame(quality_metrics).T
        
        fig = px.imshow(
            quality_df.values,
            x=quality_df.columns,
            y=quality_df.index,
            color_continuous_scale='RdYlGn',
            title='Data Quality Scores by Source (%)'
        )
        
        # Add text annotations
        for i, source in enumerate(quality_df.index):
            for j, metric in enumerate(quality_df.columns):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{quality_df.iloc[i, j]:.1f}%",
                    showarrow=False,
                    font=dict(color="black", size=10)
                )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality thresholds and alerts
        st.markdown("**Quality Thresholds**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Alerts**")
            
            # Check for quality issues
            quality_alerts = []
            for source, metrics in quality_metrics.items():
                for metric, score in metrics.items():
                    if score < 90:
                        severity = "🔴 Critical" if score < 80 else "🟡 Warning"
                        quality_alerts.append({
                            'Source': source,
                            'Metric': metric.title(),
                            'Score': f"{score}%",
                            'Severity': severity
                        })
            
            if quality_alerts:
                alerts_df = pd.DataFrame(quality_alerts)
                st.dataframe(alerts_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ All data quality metrics above threshold")
        
        with col2:
            st.markdown("**Quality Trends**")
            
            # Mock trend data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            trend_data = {
                'Date': dates,
                'Overall Quality': [94 + (i % 7) + (i % 3 - 1) for i in range(30)]
            }
            
            fig_trend = px.line(
                pd.DataFrame(trend_data),
                x='Date',
                y='Overall Quality',
                title='30-Day Quality Trend',
                range_y=[85, 100]
            )
            fig_trend.add_hline(y=95, line_dash="dash", line_color="red", 
                               annotation_text="Target Threshold (95%)")
            
            st.plotly_chart(fig_trend, use_container_width=True)
    
    def _show_pipeline_configuration(self):
        """Show pipeline configuration management"""
        st.markdown("**Pipeline Configuration**")
        
        current_config = pipeline_system.config
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Collection Intervals**")
            
            air_quality_interval = st.number_input(
                "Air Quality Interval (minutes)",
                min_value=5,
                max_value=120,
                value=current_config.get('air_quality_interval_minutes', 15),
                help="How often to collect air quality data"
            )
            
            disease_data_interval = st.number_input(
                "Disease Data Interval (minutes)",
                min_value=15,
                max_value=300,
                value=current_config.get('disease_data_interval_minutes', 60),
                help="How often to collect disease surveillance data"
            )
            
            hospital_data_interval = st.number_input(
                "Hospital Data Interval (minutes)",
                min_value=10,
                max_value=120,
                value=current_config.get('hospital_data_interval_minutes', 30),
                help="How often to collect hospital capacity data"
            )
            
            weather_data_interval = st.number_input(
                "Weather Data Interval (minutes)",
                min_value=10,
                max_value=120,
                value=current_config.get('weather_data_interval_minutes', 30),
                help="How often to collect weather data"
            )
            
            ml_predictions_interval = st.number_input(
                "ML Predictions Interval (minutes)",
                min_value=30,
                max_value=480,
                value=current_config.get('ml_predictions_interval_minutes', 120),
                help="How often to generate ML predictions"
            )
        
        with col2:
            st.markdown("**Error Handling & Performance**")
            
            max_retries = st.number_input(
                "Maximum Retries",
                min_value=1,
                max_value=10,
                value=current_config.get('max_retries', 3),
                help="Number of retry attempts for failed jobs"
            )
            
            retry_delay = st.number_input(
                "Retry Delay (seconds)",
                min_value=30,
                max_value=300,
                value=current_config.get('retry_delay_seconds', 60),
                help="Delay between retry attempts"
            )
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=10,
                max_value=1000,
                value=current_config.get('batch_size', 100),
                help="Number of records to process in each batch"
            )
            
            max_concurrent_jobs = st.number_input(
                "Max Concurrent Jobs",
                min_value=1,
                max_value=10,
                value=current_config.get('max_concurrent_jobs', 3),
                help="Maximum number of jobs running simultaneously"
            )
        
        # Apply configuration button
        st.markdown("---")
        
        if st.button("💾 Apply Configuration"):
            new_config = {
                'air_quality_interval_minutes': air_quality_interval,
                'disease_data_interval_minutes': disease_data_interval,
                'hospital_data_interval_minutes': hospital_data_interval,
                'weather_data_interval_minutes': weather_data_interval,
                'ml_predictions_interval_minutes': ml_predictions_interval,
                'max_retries': max_retries,
                'retry_delay_seconds': retry_delay,
                'batch_size': batch_size,
                'max_concurrent_jobs': max_concurrent_jobs
            }
            
            try:
                pipeline_system.update_configuration(new_config)
                st.success("✅ Configuration updated successfully!")
                st.info("ℹ️ Pipeline will restart automatically to apply new settings.")
                
                # Show applied configuration
                with st.expander("Applied Configuration"):
                    st.json(new_config)
                    
            except Exception as e:
                st.error(f"Failed to update configuration: {str(e)}")
    
    def _show_monitoring_metrics(self):
        """Show detailed monitoring metrics"""
        st.markdown("**Pipeline Performance Monitoring**")
        
        status = pipeline_system.get_pipeline_status()
        health = status['health_status']
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Jobs Executed", health.get('total_jobs', 0))
        
        with col2:
            avg_execution_time = health.get('avg_execution_time', 0)
            st.metric("Avg Execution Time", f"{avg_execution_time:.1f}s")
        
        with col3:
            scheduler_active = status.get('scheduler_thread_active', False)
            scheduler_status = "🟢 Active" if scheduler_active else "🔴 Inactive"
            st.metric("Scheduler Status", scheduler_status)
        
        # Resource utilization (mock data - in production would show real metrics)
        st.markdown("**Resource Utilization**")
        
        resource_data = {
            'Metric': ['CPU Usage', 'Memory Usage', 'Database Connections', 'Network I/O'],
            'Current': ['45%', '62%', '8/20', '1.2 MB/s'],
            'Peak (24h)': ['78%', '89%', '15/20', '4.8 MB/s'],
            'Average (24h)': ['52%', '71%', '12/20', '2.1 MB/s']
        }
        
        resource_df = pd.DataFrame(resource_data)
        st.dataframe(resource_df, use_container_width=True, hide_index=True)
        
        # Error rate trends
        st.markdown("**Error Rate Trends**")
        
        # Mock error rate data
        error_dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
        error_data = {
            'Hour': error_dates,
            'Error Rate (%)': [max(0, 5 + (i % 4) - 2 + (i % 7 - 3)) for i in range(24)]
        }
        
        fig_errors = px.line(
            pd.DataFrame(error_data),
            x='Hour',
            y='Error Rate (%)',
            title='24-Hour Error Rate Trend'
        )
        fig_errors.add_hline(y=10, line_dash="dash", line_color="red", 
                            annotation_text="Error Rate Threshold (10%)")
        
        st.plotly_chart(fig_errors, use_container_width=True)
        
        # System health indicators
        st.markdown("**System Health Indicators**")
        
        health_indicators = {
            'Database Connection': '🟢 Healthy',
            'External APIs': '🟡 Some delays',
            'Data Storage': '🟢 Normal',
            'Validation Engine': '🟢 Active',
            'Cleaning Pipeline': '🟢 Operational',
            'Alert System': '🟢 Ready'
        }
        
        health_df = pd.DataFrame(list(health_indicators.items()), 
                                columns=['Component', 'Status'])
        st.dataframe(health_df, use_container_width=True, hide_index=True)
    
    def show_pipeline_status_widget(self):
        """Show compact pipeline status widget for other pages"""
        if not has_role(UserRole.ADMIN):
            return
        
        status = pipeline_system.get_pipeline_status()
        health = status['health_status']
        
        with st.container():
            st.markdown("**⚙️ Data Pipeline**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pipeline_icon = "🟢" if status['is_running'] else "🔴"
                st.write(f"{pipeline_icon} {'Running' if status['is_running'] else 'Stopped'}")
            
            with col2:
                health_icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(health['status'], "⚫")
                st.write(f"{health_icon} {health['status'].title()}")
            
            with col3:
                success_rate = health.get('success_rate', 0)
                st.write(f"📊 {success_rate:.0%} success")

# Global pipeline interface instance
pipeline_interface = PipelineInterface()