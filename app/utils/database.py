import os
import psycopg2
from dotenv import load_dotenv
import json
from pathlib import Path
import base64
from datetime import datetime

load_dotenv()

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )

def log_detection(session_id, model_type, image_path, detections):
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            "INSERT INTO sessions (session_id, created_at) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (session_id, datetime.now())
        )
        
        for detection in detections:
            cur.execute(
                """INSERT INTO detections 
                (session_id, model_type, image_path, class, confidence, bbox, latitude, longitude) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    session_id,
                    model_type,
                    image_path,
                    detection['class'],
                    detection['confidence'],
                    json.dumps(detection['bbox']),
                    detection.get('latitude'),
                    detection.get('longitude')
                )
            )
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def get_detection_locations(species=None):
    """Get all detections with location data"""
    conn = get_connection()
    cur = conn.cursor()
    
    query = """
        SELECT class, latitude, longitude, confidence, timestamp 
        FROM detections 
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """
    params = ()
    
    if species:
        query += " AND class = %s"
        params = (species,)
    
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return [{
        "class": r[0],
        "lat": float(r[1]),
        "lon": float(r[2]),
        "confidence": float(r[3]),
        "timestamp": r[4]
    } for r in results]


def get_session_detections(session_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, session_id, model_type, image_path, class, confidence, bbox, timestamp, latitude, longitude
        FROM detections 
        WHERE session_id = %s 
        ORDER BY timestamp DESC
    """, (session_id,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return [{
        "id": r[0],
        "session_id": r[1],
        "model_type": r[2],
        "image_path": r[3],
        "class": r[4],
        "confidence": r[5],
        "bbox": r[6],
        "timestamp": r[7],
        "latitude": r[8],
        "longitude": r[9]
    } for r in results]

def get_all_sessions():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.session_id, s.created_at, COUNT(d.id) as detection_count
        FROM sessions s
        LEFT JOIN detections d ON s.session_id = d.session_id
        GROUP BY s.session_id, s.created_at
        ORDER BY s.created_at DESC
    """)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results


def create_new_session(session_id):
    """Create a new session record in the database"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (session_id,)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()


def save_model_metrics(model_name, metrics):
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        for key in ['confusion_matrix', 'precision_curve', 'recall_curve', 'f1_curve', 'p_curve', 'r_curve']:
            if key in metrics and metrics[key] and Path(metrics[key]).exists():
                with open(metrics[key], "rb") as img_file:
                    metrics[key] = base64.b64encode(img_file.read()).decode('utf-8')
        
        cur.execute(
            """INSERT INTO model_metrics (model_name, metrics) 
            VALUES (%s, %s)
            ON CONFLICT (model_name) DO UPDATE SET metrics = EXCLUDED.metrics""",
            (model_name, json.dumps(metrics))
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def get_model_metrics(model_name):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT metrics FROM model_metrics WHERE model_name = %s", (model_name,))
        result = cur.fetchone()
        return result[0] if result else None
    finally:
        cur.close()
        conn.close()

def get_aggregated_stats():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                session_id,
                model_type,
                class,
                confidence,
                timestamp
            FROM detections
            ORDER BY timestamp DESC
        """)
        results = cur.fetchall()
        
        stats = []
        for row in results:
            stats.append({
                "session_id": row[0],
                "model_type": row[1],
                "class": row[2],
                "confidence": row[3],
                "timestamp": row[4]
            })
        return stats
    finally:
        cur.close()
        conn.close()

def log_chat(session_id, user_message, bot_response):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chatbot_logs (session_id, user_message, bot_response)
            VALUES (%s, %s, %s)
        """, (session_id, user_message, bot_response))
        conn.commit()
    finally:
        cur.close()
        conn.close()
