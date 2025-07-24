import streamlit as st
from pathlib import Path
import sqlite3
import json
import shutil
import uuid
from datetime import datetime
import cv2

OFFLINE_DB = Path("data/offline_detections.db")
OFFLINE_IMAGES = Path("data/offline_images")
OFFLINE_IMAGES.mkdir(parents=True, exist_ok=True)

def init_offline_db():
    """Initialize offline SQLite database with field_observations table"""
    conn = sqlite3.connect(OFFLINE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        species TEXT NOT NULL,
        image_path TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        latitude REAL,
        longitude REAL,
        notes TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS field_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        species TEXT NOT NULL,
        image_path TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        latitude REAL,
        longitude REAL,
        notes TEXT
    )
    """)
    
    conn.commit()
    conn.close()

def save_offline_detection(detection_data):
    """Save detection to offline database and sync with main DB"""
    conn = sqlite3.connect(OFFLINE_DB)
    cursor = conn.cursor()
    
    image_filename = f"{uuid.uuid4()}.jpg"
    image_path = OFFLINE_IMAGES / image_filename
    
    if "image_array" in detection_data:
        cv2.imwrite(str(image_path), detection_data["image_array"])
    elif "image_file" in detection_data:
        with open(image_path, "wb") as f:
            f.write(detection_data["image_file"].getvalue())
    
    cursor.execute("""
    INSERT INTO detections 
    (species, image_path, confidence, bbox, latitude, longitude, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        detection_data["species"],
        str(image_path),
        detection_data["confidence"],
        json.dumps(detection_data["bbox"]),
        detection_data.get("latitude"),
        detection_data.get("longitude"),
        detection_data.get("notes", "")
    ))
    
    cursor.execute("""
    INSERT INTO field_observations 
    (session_id, species, image_path, confidence, bbox, latitude, longitude, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        detection_data.get("session_id", "offline"),
        detection_data["species"],
        str(image_path),
        detection_data["confidence"],
        json.dumps(detection_data["bbox"]),
        detection_data.get("latitude"),
        detection_data.get("longitude"),
        detection_data.get("notes", "")
    ))
    
    conn.commit()
    conn.close()

def get_offline_detections():
    """Retrieve offline detections with image handling"""
    conn = sqlite3.connect(OFFLINE_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC")
    results = cursor.fetchall()
    
    conn.close()
    
    detections = []
    for row in results:
        detection = dict(row)
        detection["image_exists"] = Path(detection["image_path"]).exists()
        detections.append(detection)
    
    return detections

def sync_field_observations(main_db_conn):
    """Sync field observations to main database"""
    offline_conn = sqlite3.connect(OFFLINE_DB)
    offline_cursor = offline_conn.cursor()
    main_cursor = main_db_conn.cursor()
    
    offline_cursor.execute("""
    SELECT * FROM field_observations 
    WHERE session_id != 'synced'
    ORDER BY timestamp DESC
    """)
    
    for row in offline_cursor.fetchall():
        row = dict(row)
        
        main_cursor.execute("""
        INSERT INTO detections 
        (session_id, model_type, image_path, class, confidence, bbox, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row["session_id"],
            "field_observation",
            row["image_path"],
            row["species"],
            row["confidence"],
            row["bbox"],
            row["timestamp"]
        ))
        
        offline_cursor.execute("""
        UPDATE field_observations 
        SET session_id = 'synced' 
        WHERE id = ?
        """, (row["id"],))
    
    offline_conn.commit()
    main_db_conn.commit()
    offline_conn.close()