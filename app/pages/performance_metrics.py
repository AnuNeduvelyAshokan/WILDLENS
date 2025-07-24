import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.database import get_model_metrics
from utils.visualization import display_base64_image
from io import BytesIO
import base64



st.set_page_config(page_title="Performance Metrics", layout="wide")
st.title("Model Performance Metrics")

# Get metrics for both models
butterfly_metrics = get_model_metrics("butterfly")
bee_metrics = get_model_metrics("bee")

# Create tabs for each model
tab1, tab2 = st.tabs(["Butterfly Model", "Bee Model"])

def display_model_metrics(metrics, model_name):
    if not metrics:
        st.warning(f"No performance data available for {model_name} model")
        return
    
    # Main metrics cards
    st.subheader("Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("mAP@0.5", f"{metrics.get('map50', 0):.2%}")
        st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
    with col2:
        st.metric("mAP@0.5:0.95", f"{metrics.get('map', 0):.2%}")
        st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
    with col3:
        st.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
        st.metric("Epochs", metrics.get('epoch', 'N/A'))
    
    st.divider()
    
    # Visualizations in expandable sections
    with st.expander("Training Curves", expanded=True):
        if "loss_curve" in metrics:
            display_base64_image(metrics["loss_curve"])
        else:
            st.warning("Training curves not available")
    
    with st.expander("Confusion Matrix", expanded=True):
        if "confusion_matrix" in metrics:
            display_base64_image(metrics["confusion_matrix"])
        else:
            st.warning("Confusion matrix not available")
    
    with st.expander("Precision-Recall Analysis", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if "pr_curve" in metrics:
                display_base64_image(metrics["pr_curve"])
            else:
                st.warning("Precision-Recall curve not available")
        with col2:
            if "f1_curve" in metrics:
                display_base64_image(metrics["f1_curve"])
            else:
                st.warning("F1 curve not available")

    with st.expander("Class Performance", expanded=True):
        if "class_metrics" in metrics:
            class_df = pd.DataFrame(metrics["class_metrics"])
            st.dataframe(
                class_df.style.format({
                    'precision': '{:.2%}',
                    'recall': '{:.2%}',
                    'f1': '{:.2%}'
                }),
                column_config={
                    "class": "Class",
                    "precision": st.column_config.NumberColumn(
                        "Precision",
                        format="%.2f",
                    ),
                    "recall": st.column_config.NumberColumn(
                        "Recall",
                        format="%.2f",
                    ),
                    "f1": st.column_config.NumberColumn(
                        "F1 Score",
                        format="%.2f",
                    )
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Visual comparison
            fig = px.bar(
                class_df.melt(id_vars="class"), 
                x="class", 
                y="value",
                color="variable",
                barmode="group",
                title=f"Class-wise Performance ({model_name})",
                labels={"value": "Score", "variable": "Metric"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Class performance data not available")

# Display each model's metrics in its tab
with tab1:
    display_model_metrics(butterfly_metrics, "Butterfly Model")

with tab2:
    display_model_metrics(bee_metrics, "Bee Model")

# Model comparison section (only shown if both models have data)
if butterfly_metrics and bee_metrics:
    st.divider()
    st.subheader("🔍 Model Comparison")
    
    comparison_data = {
        "Metric": ["mAP@0.5", "Precision", "Recall", "F1 Score", "mAP@0.5:0.95"],
        "Butterfly Model": [
            butterfly_metrics.get("map50", 0),
            butterfly_metrics.get("precision", 0),
            butterfly_metrics.get("recall", 0),
            butterfly_metrics.get("f1", 0),
            butterfly_metrics.get("map", 0)
        ],
        "Bee Model": [
            bee_metrics.get("map50", 0),
            bee_metrics.get("precision", 0),
            bee_metrics.get("recall", 0),
            bee_metrics.get("f1", 0),
            bee_metrics.get("map", 0)
        ]
    }
    
    fig = px.bar(
        pd.DataFrame(comparison_data).melt(id_vars="Metric"),
        x="Metric",
        y="value",
        color="variable",
        barmode="group",
        title="Model Performance Comparison",
        labels={"value": "Score", "variable": "Model"},
        color_discrete_map={
            "Butterfly Model": "#636EFA",
            "Bee Model": "#EF553B"
        }
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Need both models' metrics to show comparison")