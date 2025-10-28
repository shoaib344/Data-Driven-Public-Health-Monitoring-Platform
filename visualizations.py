import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import streamlit as st

class Visualizations:
    def __init__(self):
        self.color_schemes = {
            'aqi': px.colors.sequential.Reds,
            'cases': px.colors.sequential.Blues,
            'capacity': px.colors.sequential.Oranges,
            'default': px.colors.sequential.Viridis
        }
    
    def create_time_series(self, data, x, y, title, color=None, height=400):
        """Create time series visualization"""
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title=title, height=height)
            return fig
        
        if color and color in data.columns:
            fig = px.line(
                data, x=x, y=y, 
                title=title,
                color_discrete_sequence=self.color_schemes.get(color, ['#1f77b4']),
                height=height
            )
            
            # Add color coding for threshold levels
            if 'aqi' in y.lower():
                fig.add_hline(y=50, line_dash="dash", line_color="green", 
                             annotation_text="Good (0-50)")
                fig.add_hline(y=100, line_dash="dash", line_color="yellow",
                             annotation_text="Moderate (51-100)")
                fig.add_hline(y=150, line_dash="dash", line_color="orange",
                             annotation_text="Unhealthy for Sensitive (101-150)")
                fig.add_hline(y=200, line_dash="dash", line_color="red",
                             annotation_text="Unhealthy (151-200)")
        else:
            fig = px.line(data, x=x, y=y, title=title, height=height)
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=y.replace('_', ' ').title(),
            hovermode='x unified'
        )
        
        return fig
    
    def create_multi_metric_chart(self, data, metrics, title, height=500):
        """Create multi-metric visualization"""
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
        
        fig = make_subplots(
            rows=len(metrics), cols=1,
            shared_xaxes=True,
            subplot_titles=[metric.replace('_', ' ').title() for metric in metrics],
            vertical_spacing=0.08
        )
        
        for i, metric in enumerate(metrics):
            if metric in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data[metric],
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=2),
                        hovertemplate=f'{metric}: %{{y}}<extra></extra>'
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            height=height,
            title_text=title,
            showlegend=False
        )
        
        return fig
    
    def create_heatmap(self, data, x, y, z, title, height=400):
        """Create heatmap visualization"""
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
        
        # Pivot data for heatmap
        pivot_data = data.pivot_table(
            index=y, columns=x, values=z, aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlBu_r',
            hoverinfo='z'
        ))
        
        fig.update_layout(
            title=title,
            height=height,
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title()
        )
        
        return fig
    
    def create_distribution_plot(self, data, column, title, height=400):
        """Create distribution plot"""
        if data.empty or column not in data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
        
        fig = px.histogram(
            data, x=column, 
            title=title,
            nbins=30,
            marginal="box",
            height=height
        )
        
        fig.update_layout(
            xaxis_title=column.replace('_', ' ').title(),
            yaxis_title="Frequency"
        )
        
        return fig
    
    def create_correlation_heatmap(self, data, title, height=500):
        """Create correlation heatmap"""
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
        
        # Select only numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient numeric data for correlation",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
        
        correlation_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverinfo='z'
        ))
        
        fig.update_layout(
            title=title,
            height=height,
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
    
    def create_gauge_chart(self, value, title, max_value=100, thresholds=None):
        """Create gauge chart for metrics"""
        if thresholds is None:
            thresholds = [30, 60, 100]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, thresholds[0]], 'color': "lightgreen"},
                    {'range': [thresholds[0], thresholds[1]], 'color': "yellow"},
                    {'range': [thresholds[1], max_value], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': thresholds[-1]
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig
    
    def create_folium_map(self, geospatial_data):
        """Create interactive folium map"""
        if geospatial_data.empty:
            # Return a simple map centered on US
            m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
            folium.Marker([39.8283, -98.5795], 
                         popup="No data available",
                         icon=folium.Icon(color='gray')).add_to(m)
            return m
        
        # Center map on data points
        center_lat = geospatial_data['latitude'].mean()
        center_lon = geospatial_data['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
        
        # Add markers for each location
        for _, row in geospatial_data.iterrows():
            # Color based on risk level
            color_map = {
                'LOW': 'green',
                'MEDIUM': 'orange', 
                'HIGH': 'red'
            }
            
            color = color_map.get(row['risk_level'], 'blue')
            
            # Create popup content
            popup_html = f"""
            <div style="width: 200px;">
                <h4>{row['location']}</h4>
                <b>AQI:</b> {row['aqi']}<br>
                <b>Disease Cases:</b> {row['disease_cases']}<br>
                <b>Hospital Capacity:</b> {row['hospital_capacity']}%<br>
                <b>Risk Level:</b> {row['risk_level']}<br>
                <b>Population:</b> {row['population']:,}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=10,
                popup=folium.Popup(popup_html, max_width=220),
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        # Add a heatmap layer for AQI
        if 'aqi' in geospatial_data.columns:
            heat_data = [[row['latitude'], row['longitude'], row['aqi']] 
                        for _, row in geospatial_data.iterrows()]
            
            HeatMap(heat_data, 
                   min_opacity=0.2,
                   max_zoom=18,
                   blur=15,
                   name='AQI Heatmap').add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_prediction_chart(self, historical_data, predictions, title):
        """Create chart showing historical data and predictions"""
        fig = go.Figure()
        
        # Add historical data
        if not historical_data.empty:
            fig.add_trace(go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data.iloc[:, 1],  # Assuming second column is the value
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
        
        # Add predictions
        if not predictions.empty:
            fig.add_trace(go.Scatter(
                x=predictions['timestamp'],
                y=predictions.iloc[:, 1],  # Assuming second column is the prediction
                mode='lines+markers',
                name='Predicted',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Add confidence intervals if available
            if 'confidence' in predictions.columns:
                # Create confidence bands (simplified)
                upper_bound = predictions.iloc[:, 1] * 1.1
                lower_bound = predictions.iloc[:, 1] * 0.9
                
                fig.add_trace(go.Scatter(
                    x=predictions['timestamp'],
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=predictions['timestamp'],
                    y=lower_bound,
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    name='Confidence Interval',
                    fillcolor='rgba(255,0,0,0.2)',
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=500
        )
        
        return fig
