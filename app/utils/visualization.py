import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import numpy as np

def plot_detection_distribution(detections):
    if not detections:
        return
        
    df = pd.DataFrame(detections)
    
    class_counts = df['class'].value_counts().reset_index()
    class_counts.columns = ['class', 'count']

    fig = px.bar(
        class_counts,
        x='class',
        y='count',
        labels={'class': 'Class', 'count': 'Count'},
        title='<b>Class Distribution</b>',
        color='class',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Detection Count",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_confidence_distribution(detections):
    if not detections:
        return
        
    df = pd.DataFrame(detections)
    
    fig = px.histogram(
        df, 
        x='confidence', 
        nbins=20, 
        title='<b>Confidence Distribution</b>',
        color_discrete_sequence=['#636EFA']
    )
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        plot_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_confusion_matrix(cm_data):
    if not cm_data:
        return
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_data['matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=cm_data['classes'], yticklabels=cm_data['classes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    st.image(buf, use_container_width=True)

def plot_precision_recall_curve(precision, recall):
    if not precision or not recall:
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.', color='#636EFA', linewidth=2)
    plt.fill_between(recall, precision, alpha=0.2, color='#636EFA')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    st.image(buf, use_container_width=True)

def display_base64_image(base64_str):
    if not base64_str:
        return
    image_data = base64.b64decode(base64_str)
    st.image(image_data, use_container_width=True)