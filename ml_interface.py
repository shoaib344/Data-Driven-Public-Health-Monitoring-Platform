"""
Machine Learning Interface for Streamlit Public Health Monitor
Provides UI components for ML model management and predictions
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any
from advanced_ml_models import advanced_ml
from database_service import db_service
from database_models import get_db
from auth import has_role, UserRole, get_current_user_id

class MLInterface:
    """Interface for ML model management and visualization"""
    
    def __init__(self):
        self.ml_system = advanced_ml
    
    def show_ml_dashboard(self):
        """Show ML dashboard for health authorities and admins"""
        if not has_role(UserRole.HEALTH_AUTHORITY):
            st.error("Access denied. Health Authority or Admin privileges required.")
            return
        
        st.subheader("🤖 Machine Learning Dashboard")
        
        # Model management tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Model Status", "Training", "Predictions", "Performance"])
        
        with tab1:
            self._show_model_status()
        
        with tab2:
            self._show_model_training()
        
        with tab3:
            self._show_predictions_interface()
        
        with tab4:
            self._show_model_performance()
    
    def _show_model_status(self):
        """Display current model status"""
        st.markdown("**Current Model Status**")
        
        performance = self.ml_system.get_model_performance()
        
        # Model overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Models", performance['total_models'])
        
        with col2:
            st.metric("Ensemble Models", performance['ensemble_models'])
        
        with col3:
            st.metric("LSTM Models", performance['lstm_models'])
        
        with col4:
            st.metric("Metrics Covered", len(performance['metrics_covered']))
        
        # Detailed model information
        if performance['model_details']:
            st.markdown("**Model Details**")
            
            model_status_data = []
            for model_name, details in performance['model_details'].items():
                metric, model_type = model_name.rsplit('_', 1)
                status = "✅ Active" if details.get('is_fitted', False) else "❌ Not Trained"
                
                model_status_data.append({
                    'Metric': metric.upper(),
                    'Model Type': model_type.upper(),
                    'Status': status,
                    'Details': str(details.get('weights', details.get('sequence_length', 'N/A')))[:50]
                })
            
            if model_status_data:
                df = pd.DataFrame(model_status_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No models currently trained. Use the Training tab to train models.")
    
    def _show_model_training(self):
        """Interface for training ML models"""
        st.markdown("**Model Training**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Configuration**")
            
            # Location selection for training
            db = next(get_db())
            try:
                locations = db_service.health_data.get_locations(db)
                location_options = ["All Locations"] + [loc.name for loc in locations]
                
                selected_location = st.selectbox(
                    "Training Location",
                    options=location_options,
                    help="Select location for model training. 'All Locations' trains on combined data."
                )
                
                # Model types to train
                model_types = st.multiselect(
                    "Model Types",
                    options=["Ensemble", "LSTM"],
                    default=["Ensemble"],
                    help="Select which types of models to train"
                )
                
                # Training data period
                training_days = st.slider(
                    "Training Data Period (days)",
                    min_value=7,
                    max_value=90,
                    value=30,
                    help="Number of days of historical data to use for training"
                )
                
            finally:
                db.close()
        
        with col2:
            st.markdown("**Training Status**")
            
            # Check data availability
            if st.button("Check Data Availability"):
                with st.spinner("Checking data availability..."):
                    db = next(get_db())
                    try:
                        location_id = None if selected_location == "All Locations" else selected_location
                        
                        # Check data for each metric
                        metrics = ['aqi', 'disease_cases', 'bed_occupancy', 'temperature']
                        data_status = []
                        
                        end_date = datetime.utcnow()
                        start_date = end_date - timedelta(days=training_days)
                        
                        for metric in metrics:
                            df = db_service.health_data.get_health_metrics(
                                db, location_id=location_id, metric_type=metric,
                                start_date=start_date, end_date=end_date, limit=1000
                            )
                            
                            data_status.append({
                                'Metric': metric.upper(),
                                'Records': len(df),
                                'Date Range': f"{df['timestamp'].min().date() if not df.empty else 'N/A'} to {df['timestamp'].max().date() if not df.empty else 'N/A'}",
                                'Status': '✅ Ready' if len(df) >= 50 else '❌ Insufficient'
                            })
                        
                        status_df = pd.DataFrame(data_status)
                        st.dataframe(status_df, use_container_width=True)
                        
                    finally:
                        db.close()
        
        # Training execution
        st.markdown("---")
        
        if st.button("🚀 Start Model Training", type="primary"):
            if not model_types:
                st.error("Please select at least one model type to train.")
                return
            
            with st.spinner("Training models... This may take several minutes."):
                try:
                    # Get location ID
                    location_id = None
                    if selected_location != "All Locations":
                        db = next(get_db())
                        try:
                            from database_models import Location
                            location_obj = db.query(Location).filter(Location.name == selected_location).first()
                            if location_obj:
                                location_id = str(location_obj.id)
                        finally:
                            db.close()
                    
                    # Start training
                    results = self.ml_system.train_all_models(location_id)
                    
                    if results:
                        st.success("🎉 Model training completed successfully!")
                        
                        # Show training results
                        st.markdown("**Training Results**")
                        
                        for model_name, scores in results.items():
                            if isinstance(scores, dict) and scores:
                                st.write(f"**{model_name}**")
                                for metric, score in scores.items():
                                    if isinstance(score, (int, float)) and score != float('inf'):
                                        st.write(f"  • {metric}: {score:.4f}")
                    else:
                        st.warning("No models were successfully trained. Please check data availability.")
                        
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    st.error("Please check the logs for more details.")
    
    def _show_predictions_interface(self):
        """Interface for generating and viewing predictions"""
        st.markdown("**Model Predictions**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Prediction Settings**")
            
            # Location selection
            db = next(get_db())
            try:
                locations = db_service.health_data.get_locations(db)
                location_options = [loc.name for loc in locations]
                
                selected_location = st.selectbox(
                    "Prediction Location",
                    options=location_options,
                    help="Select location for predictions"
                )
                
                # Prediction horizon
                days_ahead = st.slider(
                    "Prediction Horizon (days)",
                    min_value=1,
                    max_value=14,
                    value=7,
                    help="Number of days to predict ahead"
                )
                
                # Metrics to predict
                available_metrics = list(self.ml_system.ensemble_models.keys()) + list(self.ml_system.lstm_models.keys())
                if available_metrics:
                    selected_metrics = st.multiselect(
                        "Metrics to Predict",
                        options=list(set(available_metrics)),
                        default=list(set(available_metrics))[:2],
                        help="Select which health metrics to generate predictions for"
                    )
                else:
                    st.info("No trained models available. Please train models first.")
                    selected_metrics = []
                
            finally:
                db.close()
        
        with col2:
            st.markdown("**Prediction Actions**")
            
            if st.button("📈 Generate Predictions") and selected_metrics:
                with st.spinner("Generating predictions..."):
                    try:
                        # Get location ID
                        db = next(get_db())
                        try:
                            from database_models import Location
                            location_obj = db.query(Location).filter(Location.name == selected_location).first()
                            location_id = str(location_obj.id) if location_obj else None
                        finally:
                            db.close()
                        
                        # Generate predictions
                        predictions = self.ml_system.generate_predictions(location_id, days_ahead)
                        
                        if predictions:
                            st.success(f"Generated predictions for {len(predictions)} models!")
                            
                            # Store predictions for display
                            st.session_state.ml_predictions = predictions
                            st.session_state.prediction_location = selected_location
                        else:
                            st.warning("No predictions generated. Please check model status.")
                            
                    except Exception as e:
                        st.error(f"Prediction generation failed: {str(e)}")
        
        # Display predictions if available
        if hasattr(st.session_state, 'ml_predictions') and st.session_state.ml_predictions:
            st.markdown("---")
            st.markdown("**Prediction Results**")
            
            predictions = st.session_state.ml_predictions
            location = st.session_state.get('prediction_location', 'Unknown')
            
            # Create prediction visualization tabs
            prediction_tabs = st.tabs(list(predictions.keys())[:4])  # Limit to 4 tabs
            
            for i, (model_name, pred_df) in enumerate(list(predictions.items())[:4]):
                with prediction_tabs[i]:
                    self._show_prediction_chart(pred_df, model_name, location)
    
    def _show_prediction_chart(self, pred_df: pd.DataFrame, model_name: str, location: str):
        """Display prediction chart"""
        if pred_df.empty:
            st.info("No prediction data available")
            return
        
        # Create prediction chart
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=pred_df['timestamp'],
            y=pred_df['predicted_value'],
            mode='lines',
            name='Predicted',
            line=dict(color='blue', width=2)
        ))
        
        # Add confidence bands if available
        if 'confidence_lower' in pred_df.columns and 'confidence_upper' in pred_df.columns:
            fig.add_trace(go.Scatter(
                x=pred_df['timestamp'].tolist() + pred_df['timestamp'][::-1].tolist(),
                y=pred_df['confidence_upper'].tolist() + pred_df['confidence_lower'][::-1].tolist(),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title=f"{model_name} Predictions - {location}",
            xaxis_title="Date",
            yaxis_title="Predicted Value",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_pred = pred_df['predicted_value'].mean()
            st.metric("Average Prediction", f"{avg_pred:.2f}")
        
        with col2:
            if 'confidence_score' in pred_df.columns:
                avg_confidence = pred_df['confidence_score'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.2%}")
        
        with col3:
            pred_range = pred_df['predicted_value'].max() - pred_df['predicted_value'].min()
            st.metric("Prediction Range", f"{pred_range:.2f}")
    
    def _show_model_performance(self):
        """Show model performance metrics"""
        st.markdown("**Model Performance Analysis**")
        
        # Get database predictions for analysis
        db = next(get_db())
        try:
            # Get recent predictions from database
            predictions_df = db_service.ml_models.get_predictions(db, hours_ahead=168)  # Last week
            
            if predictions_df.empty:
                st.info("No prediction data available for performance analysis. Generate some predictions first.")
                return
            
            # Performance analysis
            st.markdown("**Recent Prediction Activity**")
            
            # Group by metric type
            metrics_summary = predictions_df.groupby('metric_type').agg({
                'predicted_value': ['count', 'mean', 'std'],
                'confidence_score': 'mean'
            }).round(2)
            
            if not metrics_summary.empty:
                # Flatten column names
                metrics_summary.columns = ['Count', 'Mean Value', 'Std Dev', 'Avg Confidence']
                metrics_summary = metrics_summary.reset_index()
                
                st.dataframe(metrics_summary, use_container_width=True)
                
                # Prediction confidence distribution
                st.markdown("**Prediction Confidence Distribution**")
                
                fig = px.histogram(
                    predictions_df,
                    x='confidence_score',
                    title='Distribution of Prediction Confidence Scores',
                    nbins=20
                )
                fig.update_xaxis(title="Confidence Score")
                fig.update_yaxis(title="Frequency")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No summary data available.")
                
        finally:
            db.close()
    
    def show_public_predictions(self, data_sources):
        """Show ML predictions for public users"""
        st.subheader("🔮 Health Predictions")
        st.markdown("AI-powered forecasts for health metrics in your area")
        
        # Get user's preferred locations
        user_id = get_current_user_id()
        if not user_id:
            st.info("Please log in to see personalized predictions.")
            return
        
        db = next(get_db())
        try:
            user_locations = db_service.preferences.get_user_locations(db, user_id)
            
            if not user_locations:
                st.info("Add locations to your preferences to see AI predictions.")
                return
            
            # Show predictions for user locations
            for location in user_locations[:2]:  # Show for top 2 locations
                st.markdown(f"**📍 {location.name} - AI Predictions**")
                
                # Get recent predictions from database
                location_predictions = db_service.ml_models.get_predictions(
                    db, location_id=str(location.id), hours_ahead=72  # Next 3 days
                )
                
                if not location_predictions.empty:
                    # Show prediction summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Air quality prediction if available
                        aqi_preds = location_predictions[location_predictions['metric_type'] == 'aqi']
                        if not aqi_preds.empty:
                            next_aqi = aqi_preds.iloc[0]['predicted_value']
                            aqi_confidence = aqi_preds.iloc[0]['confidence_score']
                            
                            st.metric(
                                "Next Day AQI Forecast",
                                f"{next_aqi:.0f}",
                                help=f"Confidence: {aqi_confidence:.0%}"
                            )
                    
                    with col2:
                        # Disease cases prediction if available
                        disease_preds = location_predictions[location_predictions['metric_type'] == 'disease_cases']
                        if not disease_preds.empty:
                            next_cases = disease_preds.iloc[0]['predicted_value']
                            cases_confidence = disease_preds.iloc[0]['confidence_score']
                            
                            st.metric(
                                "Disease Cases Trend",
                                f"{next_cases:.0f}",
                                help=f"Confidence: {cases_confidence:.0%}"
                            )
                    
                    # Show trend chart for most confident prediction
                    best_metric = location_predictions.loc[location_predictions['confidence_score'].idxmax()]
                    metric_data = location_predictions[location_predictions['metric_type'] == best_metric['metric_type']]
                    
                    if len(metric_data) > 1:
                        fig = px.line(
                            metric_data,
                            x='timestamp',
                            y='predicted_value',
                            title=f"3-Day {best_metric['metric_type'].upper()} Forecast",
                            height=300
                        )
                        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No AI predictions available for {location.name}. Predictions are generated by health authorities.")
                
                st.markdown("---")
                
        finally:
            db.close()

# Global ML interface instance
ml_interface = MLInterface()