import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.database import get_aggregated_stats

import numpy as np

st.set_page_config(page_title="Detection Analytics", layout="wide")
st.title("Detection Analytics")

# Get aggregated stats
stats = get_aggregated_stats()

if not stats:
    st.warning("No analytics data available yet. Perform detections first!")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(stats)

if df.empty:
    st.warning("No data available for analytics")
    st.stop()

# Convert timestamp to datetime
df['date'] = pd.to_datetime(df['timestamp']).dt.date
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_name'] = pd.to_datetime(df['timestamp']).dt.day_name()
model_map = {
    "butterfly": "Butterfly",
    "bee": "Bee",
    "fast_rcnn": "Fast R-CNN"
}
df['model_name'] = df['model_type'].map(model_map)

# Show summary cards
st.header("Global Detection Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Detections", len(df))
col2.metric("Unique Sessions", df['session_id'].nunique())
col3.metric("Most Active Model", df['model_name'].mode()[0])
col4.metric("Average Confidence", f"{df['confidence'].mean():.2%}")

# Create tabs for different analytics views
tab1, tab2, tab3, tab4 = st.tabs([
    "Temporal Analysis", 
    "Model Performance", 
    "Class Distribution", 
    "Session Insights"
])

with tab1:
    st.subheader("Detection Activity Over Time")
    
    # Daily detection count
    daily_counts = df.groupby('date').size().reset_index(name='counts')
    fig1 = px.line(
        daily_counts, 
        x='date', 
        y='counts',
        title='<b>Daily Detection Activity</b>',
        markers=True
    )
    fig1.update_layout(height=300)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Hourly detection pattern
    hourly_counts = df.groupby('hour').size().reset_index(name='counts')
    fig2 = px.bar(
        hourly_counts, 
        x='hour', 
        y='counts',
        title='<b>Hourly Detection Pattern</b>',
        color='counts',
        color_continuous_scale='Blues'
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Day of week pattern
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df.groupby('day_name').size().reindex(day_order).reset_index(name='counts')
    fig3 = px.bar(
        day_counts, 
        x='day_name', 
        y='counts',
        title='<b>Detections by Day of Week</b>',
        color='counts',
        color_continuous_scale='Greens'
    )
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Detection count by model
        model_counts = df.groupby('model_name').size().reset_index(name='counts')
        fig1 = px.pie(
            model_counts, 
            values='counts', 
            names='model_name',
            title='<b>Detections by Model</b>',
            hole=0.4
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Confidence by model
        fig2 = px.box(
            df, 
            x='model_name', 
            y='confidence',
            title='<b>Confidence Distribution by Model</b>',
            color='model_name'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Model performance over time
    model_daily = df.groupby(['date', 'model_name']).size().reset_index(name='counts')
    fig3 = px.line(
        model_daily, 
        x='date', 
        y='counts', 
        color='model_name',
        title='<b>Daily Detection Activity by Model</b>',
        markers=True
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Class Distribution Analysis")
    
    # Top classes detected
    top_classes = df['class'].value_counts().nlargest(10).reset_index()
    top_classes.columns = ['class', 'count']  # Rename for clarity

    fig1 = px.bar(
        top_classes, 
        x='class', 
        y='count',
        title='<b>Top 10 Detected Classes</b>',
        labels={'class': 'Class', 'count': 'Count'},
        color='class',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    st.plotly_chart(fig1, use_container_width=True)
    
    # Confidence by class
    class_confidence = df.groupby('class')['confidence'].mean().reset_index()
    class_confidence = class_confidence.sort_values('confidence', ascending=False)
    fig2 = px.bar(
        class_confidence, 
        x='class', 
        y='confidence',
        title='<b>Average Confidence by Class</b>',
        labels={'confidence': 'Average Confidence'},
        color='confidence',
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.subheader("Session Analysis")
    
    # Session activity
    session_activity = df.groupby('session_id').agg(
        detections=('class', 'count'),
        unique_classes=('class', 'nunique'),
        avg_confidence=('confidence', 'mean'),
        first_detection=('timestamp', 'min'),
        last_detection=('timestamp', 'max')
    ).reset_index()
    
    # Calculate session duration
    session_activity['duration'] = (
        session_activity['last_detection'] - session_activity['first_detection']
    ).dt.total_seconds() / 60  # in minutes
    
    # Display session metrics
    st.dataframe(
        session_activity.sort_values('detections', ascending=False),
        column_config={
            "session_id": "Session ID",
            "detections": "Detection Count",
            "unique_classes": "Unique Classes",
            "avg_confidence": st.column_config.NumberColumn(
                "Avg Confidence",
                format="%.2f",
            ),
            "first_detection": "First Detection",
            "last_detection": "Last Detection",
            "duration": "Duration (min)"
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Session duration vs detections
    fig = px.scatter(
        session_activity,
        x='duration',
        y='detections',
        size='unique_classes',
        color='avg_confidence',
        hover_name='session_id',
        title='<b>Session Activity Analysis</b>',
        labels={
            'duration': 'Session Duration (minutes)',
            'detections': 'Number of Detections',
            'unique_classes': 'Unique Classes',
            'avg_confidence': 'Average Confidence'
        }
    )
    st.plotly_chart(fig, use_container_width=True)