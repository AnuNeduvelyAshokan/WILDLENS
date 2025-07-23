import streamlit as st
from utils.detection import load_model, predict_image
from utils.database import log_detection, get_all_sessions, get_session_detections, create_new_session
from utils.visualization import plot_detection_distribution, plot_confidence_distribution
from utils.species_map import show_species_map
from utils.offline_mode import init_offline_db, save_offline_detection, get_offline_detections, sync_field_observations
from utils.chatbot_core import generate_answer, load_model as load_chatbot_model
import os
import uuid
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import pandas as pd
import sqlite3
import streamlit.components.v1 as components
import base64
import json
import sys
import warnings
import plotly.express as px

init_offline_db()

SPECIES_LOCATIONS = {
    "Danaus plexippus": {
        "points": [
            {"lat": 39.8283, "lon": -98.5795, "count": 15},  # North America
            {"lat": 19.4326, "lon": -99.1332, "count": 8},   # Mexico
            {"lat": -33.8688, "lon": 151.2093, "count": 3}    # Australia
        ],
        "threat_status": "Endangered"
    },
    "Osmia cornuta": {
        "points": [
            {"lat": 45.9432, "lon": 24.9668, "count": 22},    # Europe
            {"lat": 41.9028, "lon": 12.4964, "count": 18}     # Mediterranean
        ],
        "threat_status": "Least Concern"
    },
    "Heliconius charithonia": {
        "points": [
            {"lat": 25.8, "lon": -80.2, "count": 12},  # southern Florida, USA
            {"lat": 29.8, "lon": -95.4, "count": 10},  # southern Texas, USA
            {"lat": 19.0, "lon": -100.1, "count": 8}   # central Mexico
    ],
    "threat_status": "G5 (Secure)"
    },
    "Battus philenor": {
        "points": [
            {"lat": 34.0, "lon": -118.2, "count": 14}, # Los Angeles area, USA
            {"lat": 30.5, "lon": -91.2, "count": 11},  # Louisiana, USA
            {"lat": 29.1, "lon": -94.8, "count": 7}    # Gulf Coast, Texas, USA
    ],
    "threat_status": "Least Concern"
    },
    "Eurytides marcellus": {
        "points": [
            {"lat": 39.3, "lon": -76.6, "count": 13},  # Maryland, USA
            {"lat": 36.2, "lon": -86.7, "count": 10},  # Tennessee, USA
            {"lat": 45.4, "lon": -73.6, "count": 6}    # Quebec, Canada
        ],
        "threat_status": "Secure (NatureServe G5)"
    },
    "Speyeria cybele": {
        "points": [
            {"lat": 44.0, "lon": -92.5, "count": 9},   # Minnesota, USA
            {"lat": 40.0, "lon": -105.3, "count": 7},  # Colorado, USA
            {"lat": 49.2, "lon": -123.1, "count": 5}   # British Columbia, Canada
        ],
        "threat_status": "Least Concern"
    },
    "Junonia coenia": {
        "points": [
            {"lat": 33.7, "lon": -84.4, "count": 16},  # Atlanta, Georgia, USA
            {"lat": 29.8, "lon": -95.4, "count": 12},  # Houston, Texas, USA
            {"lat": 40.7, "lon": -74.0, "count": 8}    # New York City, USA
        ],
        "threat_status": "Least Concern"
    },
    "Colias eurytheme": {
        "points": [
            {"lat": 41.9, "lon": -87.6, "count": 14},  # Chicago, Illinois, USA
            {"lat": 38.9, "lon": -77.0, "count": 11},  # Washington, D.C., USA
            {"lat": 34.0, "lon": -118.2, "count": 9}   # Los Angeles, California, USA
        ],
        "threat_status": "Least Concern"
    },
    "Anartia jatrophae": {
        "points": [
            {"lat": 25.8, "lon": -80.3, "count": 15},  # Miami, Florida, USA
            {"lat": 22.2, "lon": -79.4, "count": 12},  # The Bahamas
            {"lat": 18.5, "lon": -69.9, "count": 7}    # Dominican Republic
        ],
        "threat_status": "Least Concern"
    }
}

MODEL_PATHS = {
    "butterfly": "models/butterfly_detection",
    "bee": "models/bee_detection",
    "fast_rcnn": "models/animal_detection",
    "chatbot": "models/hugging_face_t5/species-t5-finetuned/checkpoint-360"
}

if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = "butterfly"
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    create_new_session(st.session_state.session_id)
if 'detection_data' not in st.session_state:
    st.session_state.detection_data = None
if 'offline_mode' not in st.session_state:
    st.session_state.offline_mode = False
if 'chatbot_model' not in st.session_state:
    st.session_state.chatbot_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'species' not in st.session_state:
    st.session_state.species = ""

st.set_page_config(
    page_title="Butterfly & Bee Conservation System",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

def chatbot_ui():
    st.markdown("<h2>Educational Chatbot</h2>", unsafe_allow_html=True)
    st.markdown("Ask questions about any detected animal species.")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.species = st.text_input(
            "Species (e.g. Panthera leo)", 
            value=st.session_state.species,
            key="species_input"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.species = ""
            st.rerun()

    question = st.text_input("Ask a question:", key="question_input")

    if st.button("Ask", type="primary") and question and st.session_state.species:
        if st.session_state.chatbot_model:
            model, tokenizer, device = st.session_state.chatbot_model
            answer = generate_answer(question, st.session_state.species, model, tokenizer, device)
            st.session_state.chat_history.append(("user", question))
            st.session_state.chat_history.append(("bot", answer))
            st.rerun()
        else:
            st.error("Chatbot model not loaded properly")

    st.markdown("### Chat History")
    chat_container = st.container()
    with chat_container:
        for speaker, msg in st.session_state.chat_history:
            bubble = f"""
            <div style='display: flex; justify-content: {"flex-start" if speaker=="user" else "flex-end"}; margin: 8px 0;'>
                <div style='max-width: 70%; background-color: {"#dcf8c6" if speaker=="user" else "#f1f0f0"};
                            padding: 10px 15px; border-radius: 15px; font-size: 16px; color: black;'>
                    {msg}
                </div>
            </div>
            """
            st.markdown(bubble, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=80)
    st.title("Wild Life Conservation System")
    
    # System mode toggle
    st.subheader("System Mode")
    st.session_state.offline_mode = st.checkbox(
        "Offline Field Mode", 
        value=st.session_state.offline_mode,
        help="Enable for field work without internet"
    )
    
    st.divider()
    st.subheader("Model Configuration")
    model_type = st.selectbox(
        "Select Your Model:",
        ["Butterfly Detection", "Bee Detection", "Animal Detection", "QA-Chatbot Assistant"],
        index=0 if st.session_state.model_type == "butterfly" else 
            1 if st.session_state.model_type == "bee" else
            2 if st.session_state.model_type == "fast_rcnn" else 3,
        disabled=st.session_state.offline_mode
    )

    model_type_key = {
        "Butterfly Detection": "butterfly",
        "Bee Detection": "bee",
        "Animal Detection": "fast_rcnn",
        "QA-Chatbot Assistant": "chatbot"
    }[model_type]
    
    if st.button("Load Model", use_container_width=True, type="primary"):
        try:
            if model_type_key == "chatbot":
                st.session_state.chatbot_model = load_chatbot_model(MODEL_PATHS["chatbot"])
                st.session_state.model_type = model_type_key
                st.success("QA-Chatbot Assistant Model Loaded Successfully!")
            else:
                st.session_state.model = load_model(
                    model_type=model_type_key,
                    model_path=MODEL_PATHS[model_type_key]
                )
                st.session_state.model_type = model_type_key
                st.success(f"{model_type} Model Loaded Successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.session_state.model = None
    
    st.divider()
    st.subheader("Session Management")
    
    if st.button("New Session", use_container_width=True, key="new_session_btn"):
        st.session_state.session_id = str(uuid.uuid4())
        create_new_session(st.session_state.session_id)
        st.session_state.detection_data = None
        st.success(f"New session created: {st.session_state.session_id}")
        st.rerun()
    
    sessions = get_all_sessions()
    session_options = [s[0] for s in sessions] if sessions else [st.session_state.session_id]
    
    selected_session = st.selectbox(
        "Active Session", 
        session_options,
        index=session_options.index(st.session_state.session_id) if st.session_state.session_id in session_options else 0,
        disabled=st.session_state.offline_mode
    )
    
    if selected_session != st.session_state.session_id:
        st.session_state.session_id = selected_session
        st.session_state.detection_data = None
        st.rerun()
    
    st.info(f"**Session ID:** {st.session_state.session_id}")
    
    st.divider()
    st.subheader("System Status")
    st.caption(f"Models loaded: {'Loaded Successfully!' if st.session_state.model or st.session_state.chatbot_model else 'Not Loaded'}")
    st.caption(f"Active model: {model_type}")
    st.caption(f"Detections in session: {len(get_session_detections(st.session_state.session_id))}")
    if st.session_state.offline_mode:
        st.warning("Offline mode active - data saved locally")

st.title("Wild Life Detection System")
st.caption("Advanced Monitoring for Species Detection and Conservation")

if st.session_state.model_type == "chatbot":
    chatbot_ui()
else:
    tab_names = [
        "Image Detection", 
        "Detection History", 
        "Species Map", 
        "Field Observations"
    ] if not st.session_state.offline_mode else [
        "Field Detection",
        "Field Observations"
    ]

    tabs = st.tabs(tab_names)

    with tabs[0]:
        if st.session_state.offline_mode:
            st.header("Field Detection Mode")
            st.warning("Internet-dependent features disabled")
        else:
            st.header("Image Detection")
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.5, 0.01,
            key="conf_threshold"
        )
        
        col_upload, col_stats = st.columns([2, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload an image for detection:", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None and st.session_state.model:
                try:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    detections, annotated_img = predict_image(
                        st.session_state.model, 
                        image_rgb,
                        conf=confidence_threshold,
                        imgsz=640
                    )
                    
                    st.session_state.detection_data = {
                        "original": image_rgb,
                        "annotated": annotated_img,
                        "detections": detections,
                        "image_file": uploaded_file
                    }
                    
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
        
        with col_stats:
            st.subheader("Detection Statistics")
            if st.session_state.detection_data:
                detections = st.session_state.detection_data["detections"]
                if detections:
                    num_detections = len(detections)
                    classes = set([d['class'] for d in detections])
                    avg_confidence = sum(d['confidence'] for d in detections) / num_detections
                    
                    st.metric("Objects Detected", num_detections)
                    st.metric("Classes Identified", len(classes))
                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
                    
                    if not st.session_state.offline_mode:
                        class_counts = {cls: sum(1 for d in detections if d['class'] == cls) for cls in classes}
                        fig = px.bar(
                            x=list(class_counts.keys()),
                            y=list(class_counts.values()),
                            labels={'x': 'Class', 'y': 'Count'},
                            title="Class Distribution"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No objects detected in this image")
            else:
                st.info("Upload an image to see detection statistics")
    
        if st.session_state.detection_data:
            st.divider()
            st.subheader("Detection Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Original Image")
                st.image(st.session_state.detection_data["original"], use_container_width=True)
            
            with col2:
                st.caption("Detected Objects")
                st.image(st.session_state.detection_data["annotated"], use_container_width=True)
            
            if st.session_state.detection_data["detections"]:
                st.subheader("Detection Details")
                
                if st.session_state.offline_mode:
                    for detection in st.session_state.detection_data["detections"]:
                        save_offline_detection({
                            "species": detection["class"],
                            "image_file": st.session_state.detection_data["image_file"],
                            "confidence": detection["confidence"],
                            "bbox": detection["bbox"],
                            "latitude": None,
                            "longitude": None,
                            "notes": "Field observation",
                            "session_id": st.session_state.session_id
                        })
                    st.success(f"Saved {len(st.session_state.detection_data['detections'])} detections to offline storage")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        cv2.imwrite(tmp_file.name, cv2.cvtColor(
                            st.session_state.detection_data["annotated"], 
                            cv2.COLOR_RGB2BGR
                        ))
                        
                        for detection in st.session_state.detection_data["detections"]:
                            log_detection(
                                st.session_state.session_id,
                                st.session_state.model_type,
                                tmp_file.name,
                                [detection]
                            )
                    st.success("Detections saved to database")
                
                detection_df = pd.DataFrame(st.session_state.detection_data["detections"])
                detection_df = detection_df[['class', 'confidence', 'bbox']]
                detection_df['confidence'] = detection_df['confidence'].apply(lambda x: f"{x:.2%}")
                st.dataframe(
                    detection_df,
                    column_config={
                        "class": "Class",
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        ),
                        "bbox": "Bounding Box"
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                if not st.session_state.offline_mode:
                    st.subheader("Detection Analysis")
                    plot_detection_distribution(st.session_state.detection_data["detections"])
                    plot_confidence_distribution(st.session_state.detection_data["detections"])
            else:
                st.warning("No objects detected in this image")

    if not st.session_state.offline_mode and len(tabs) > 1:
        with tabs[1]:
            st.header("Detection History")
            
            sessions = get_all_sessions()
            if sessions:
                selected_session = st.selectbox(
                    "Select Session",
                    [s[0] for s in sessions],
                    index=next((i for i, s in enumerate(sessions) if s[0] == st.session_state.session_id), 0),
                    key="session_select"
                )
                
                detections = get_session_detections(selected_session)
                
                if detections:
                    st.info(f"Showing {len(detections)} detections for session: {selected_session}")
                    
                    session_df = pd.DataFrame(detections)
                    if not session_df.empty:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Detections", len(session_df))
                        col2.metric("Unique Classes", session_df['class'].nunique())
                        col3.metric("Avg Confidence", f"{session_df['confidence'].mean():.2%}")
                    
                    for i, detection in enumerate(detections):
                        with st.expander(f"Detection {i+1} - {detection['class']} ({detection['confidence']:.2%}) - {detection['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if Path(detection['image_path']).exists():
                                    st.image(detection['image_path'], caption="Detected Image")
                                else:
                                    st.warning("Image file not available")
                            with col2:
                                model_labels = {
                                    "butterfly": "Butterfly Detection",
                                    "bee": "Bee Detection",
                                    "fast_rcnn": "Animal Detection"
                                }
                                model_label = model_labels.get(detection["model_type"], "Unknown Model")
                                
                                st.write(f"**Session:** {detection['session_id']}")
                                st.write(f"**Model:** {model_label}")
                                st.write(f"**Detected Class:** {detection['class']}")
                                st.write(f"**Confidence:** {detection['confidence']:.2%}")
                                st.write(f"**Bounding Box:** {detection['bbox']}")
                                st.write(f"**Timestamp:** {detection['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.warning(f"No detections found for session: {selected_session}")
            else:
                st.info("No detection history available")

    if not st.session_state.offline_mode and len(tabs) > 2:
        with tabs[2]:
            st.header("Species Distribution Map")
            
            species_options = list(SPECIES_LOCATIONS.keys()) + ["All Species"]
            selected_species = st.selectbox(
                "Select Species to View",
                species_options,
                index=len(species_options)-1,
                key="species_select"
            )
            
            if selected_species == "All Species":
                all_points = []
                for species, data in SPECIES_LOCATIONS.items():
                    all_points.extend(data["points"])
                show_species_map({"points": all_points})
            else:
                show_species_map(SPECIES_LOCATIONS[selected_species])

    if len(tabs) > (3 if not st.session_state.offline_mode else 1):
        with tabs[-1]:
            st.header("Field Observations")
            
            if st.session_state.offline_mode:
                st.success("You're in offline mode. All observations are saved locally.")
            else:
                st.info("Viewing field observations from offline mode")
            
            offline_detections = get_offline_detections()
            if offline_detections:
                st.subheader("Saved Field Detections")
                
                for detection in offline_detections:
                    with st.expander(f"{detection['species']} ({detection['confidence']:.2%}) - {detection['timestamp']}"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.write(f"**Species:** {detection['species']}")
                            st.write(f"**Confidence:** {detection['confidence']:.2%}")
                            if detection['latitude'] and detection['longitude']:
                                st.write(f"**Location:** {detection['latitude']}, {detection['longitude']}")
                            st.write(f"**Notes:** {detection['notes']}")
                        
                        with col2:
                            if detection['image_exists']:
                                st.image(detection['image_path'], caption="Field observation")
                            else:
                                st.warning("Image file missing")
                
                if not st.session_state.offline_mode and st.button("Sync to Database"):
                    with st.spinner("Syncing field observations..."):
                        try:
                            sync_field_observations()
                            st.success("Field observations synced to main database!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Sync failed: {str(e)}")
            else:
                st.info("No field observations recorded yet")