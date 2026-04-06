import cv2
import mediapipe as mp
from scipy.spatial import distance
import numpy as np
from flask import Flask, render_template, Response, jsonify
import threading
import pygame
import pyttsx3
import time
import csv
import os
from datetime import datetime

# Initialize pygame mixer for audio alerts
pygame.mixer.init()

app = Flask(__name__)

# Detection state
detection_state = {
    "running": False,
    "ear": 0.0,
    "mar": 0.0,
    "status": "Stopped",
    "drowsy_alert": False,
    "yawn_alert": False,
    "drowsy_frames": 0,
    "yawn_frames": 0,
    # Session statistics for dashboard
    "total_frames": 0,
    "drowsy_total_frames": 0,
    "session_start_time": None
}

# MediaPipe face landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [61, 291, 39, 0, 269, 405, 13, 14]

# Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
FRAME_CHECK = 60  # ~2 seconds at 30fps

# Global objects
cap = None
face_mesh = None
output_frame = None
lock = threading.Lock()
alert_lock = threading.Lock()
last_alert_time = 0
csv_file = None
csv_writer = None

# --- 6.4 Alert & Data Logging System ---
def speak_alert(message="Please stay awake!"):
    """Speak voice alert in background thread"""
    global last_alert_time
    
    with alert_lock:
        current_time = time.time()
        if current_time - last_alert_time < 3:  # Prevent spam (3 sec cooldown)
            return
        last_alert_time = current_time
    
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speed
            engine.setProperty('volume', 1.0)  # Volume
            engine.say(message)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"Voice alert error: {e}")
    
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()

# --- 6.2 EAR and MAR Calculation (Geometry) ---
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_points):
    A = distance.euclidean(mouth_points[2], mouth_points[6])
    B = distance.euclidean(mouth_points[3], mouth_points[7])
    C = distance.euclidean(mouth_points[4], mouth_points[5])
    D = distance.euclidean(mouth_points[0], mouth_points[1])
    return (A + B + C) / (3.0 * D)

# --- 6.1 Video Stream Management (Multi-threading) ---
def process_frames():
    global cap, face_mesh, output_frame, detection_state, csv_file, csv_writer
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    
    # Setup CSV logging - single file, append mode
    csv_filename = "detection_log.csv"
    file_exists = os.path.exists(csv_filename)
    csv_file = open(csv_filename, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(['Timestamp', 'EAR', 'MAR', 'Face_Detected', 'Drowsy_Alert', 'Yawn_Alert', 'Event'])
    print(f"Logging data to: {csv_filename} (append mode)")
    
    # Logging control
    frame_count = 0
    LOG_INTERVAL = 30  # Log every 30 frames (~1 per second)
    prev_drowsy_alert = False
    prev_yawn_alert = False
    
    while detection_state["running"]:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        ear = 0.0
        mar = 0.0
        
        if results.multi_face_landmarks:
            detection_state["status"] = "Face Detected"
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # Extract eye coordinates
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_INDICES]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_INDICES]
            mouth = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH_INDICES]
            
            # Calculate ratios
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            # Draw landmarks
            for p in left_eye + right_eye:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)
            for p in mouth:
                cv2.circle(frame, p, 2, (255, 0, 255), -1)
            
            # --- 6.3 Drowsiness Logic Engine (State Machine) ---
            # Drowsiness detection
            if ear < EAR_THRESHOLD:
                detection_state["drowsy_frames"] += 1
                if detection_state["drowsy_frames"] >= FRAME_CHECK:
                    if not detection_state["drowsy_alert"]:  # Only trigger once
                        detection_state["drowsy_alert"] = True
                        speak_alert("Please stay awake!")
                        # Play sound alert
                        try:
                            pygame.mixer.music.load("alert.wav")
                            pygame.mixer.music.play()
                        except:
                            try:
                                import winsound
                                winsound.Beep(1000, 500)
                            except:
                                print("BEEP! DROWSINESS ALERT!")
                    cv2.putText(frame, "DROWSY!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                detection_state["drowsy_frames"] = 0
                detection_state["drowsy_alert"] = False
            
            # Yawn detection
            if mar > MAR_THRESHOLD:
                detection_state["yawn_frames"] += 1
                if detection_state["yawn_frames"] >= FRAME_CHECK // 2:
                    if not detection_state["yawn_alert"]:  # Only trigger once
                        detection_state["yawn_alert"] = True
                        speak_alert("You are yawning, please take a break!")
                        # Play sound alert
                        try:
                            pygame.mixer.music.load("alert.wav")
                            pygame.mixer.music.play()
                        except:
                            try:
                                import winsound
                                winsound.Beep(800, 500)
                            except:
                                print("BEEP! YAWN ALERT!")
                    cv2.putText(frame, "YAWNING!", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            else:
                detection_state["yawn_frames"] = 0
                detection_state["yawn_alert"] = False
        else:
            detection_state["status"] = "No Face"
            detection_state["drowsy_frames"] = 0
            detection_state["yawn_frames"] = 0
        
        detection_state["ear"] = round(ear, 3)
        detection_state["mar"] = round(mar, 3)
        
        # Track session statistics for dashboard
        detection_state["total_frames"] += 1
        if detection_state["drowsy_alert"]:
            detection_state["drowsy_total_frames"] += 1
        
        # CSV Logging
        frame_count += 1
        should_log = False
        event = "periodic"
        face_detected = detection_state["status"] == "Face Detected"
        
        # Log on alert state changes
        if detection_state["drowsy_alert"] != prev_drowsy_alert:
            should_log = True
            event = "DROWSY_START" if detection_state["drowsy_alert"] else "DROWSY_END"
            prev_drowsy_alert = detection_state["drowsy_alert"]
        
        if detection_state["yawn_alert"] != prev_yawn_alert:
            should_log = True
            event = "YAWN_START" if detection_state["yawn_alert"] else "YAWN_END"
            prev_yawn_alert = detection_state["yawn_alert"]
        
        # Log every N frames
        if frame_count >= LOG_INTERVAL:
            should_log = True
            frame_count = 0
        
        if should_log and csv_writer:
            csv_writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                round(ear, 4),
                round(mar, 4),
                face_detected,
                detection_state["drowsy_alert"],
                detection_state["yawn_alert"],
                event
            ])
            csv_file.flush()  # Ensure data is written
        
        # Display info on frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, detection_state["status"], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        with lock:
            output_frame = frame.copy()
    
    # Cleanup
    if csv_file:
        csv_file.close()
        print("CSV file closed.")
    cap.release()
    detection_state["status"] = "Stopped"

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 6.7 Session Control Logic ---
@app.route('/start')
def start_detection():
    if not detection_state["running"]:
        detection_state["running"] = True
        # Reset session statistics
        detection_state["total_frames"] = 0
        detection_state["drowsy_total_frames"] = 0
        detection_state["session_start_time"] = time.time()
        thread = threading.Thread(target=process_frames, daemon=True)
        thread.start()
    return jsonify({"status": "started"})

@app.route('/stop')
def stop_detection():
    detection_state["running"] = False
    return jsonify({"status": "stopped"})

# --- 6.5 Real-Time Statistics Calculation ---
@app.route('/metrics')
def get_metrics():
    # Calculate drowsiness percentage
    total = detection_state["total_frames"]
    drowsy_pct = 0.0
    if total > 0:
        drowsy_pct = round((detection_state["drowsy_total_frames"] / total) * 100, 1)
    
    # Calculate session duration
    duration = 0
    if detection_state["session_start_time"]:
        duration = int(time.time() - detection_state["session_start_time"])
    
    return jsonify({
        "ear": detection_state["ear"],
        "mar": detection_state["mar"],
        "status": detection_state["status"],
        "drowsy": detection_state["drowsy_alert"],
        "yawn": detection_state["yawn_alert"],
        "drowsy_percentage": drowsy_pct,
        "session_duration": duration
    })

# --- 6.6 Historical Data Analysis ---
@app.route('/history')
def get_history():
    """Read and parse detection_log.csv for historical data dashboard"""
    csv_filename = "detection_log.csv"
    
    if not os.path.exists(csv_filename):
        return jsonify({
            "error": "No history file found",
            "data": [],
            "summary": {}
        })
    
    try:
        data = []
        total_rows = 0
        drowsy_rows = 0
        yawn_rows = 0
        events = []
        ear_values = []
        mar_values = []
        
        with open(csv_filename, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Get last 500 rows for display
            recent_rows = rows[-500:] if len(rows) > 500 else rows
            
            for row in rows:
                total_rows += 1
                if row.get('Drowsy_Alert') == 'True':
                    drowsy_rows += 1
                if row.get('Yawn_Alert') == 'True':
                    yawn_rows += 1
                    
                # Collect event rows
                event = row.get('Event', '')
                if event and event != 'periodic':
                    events.append({
                        'timestamp': row.get('Timestamp', ''),
                        'event': event,
                        'ear': float(row.get('EAR', 0)),
                        'mar': float(row.get('MAR', 0))
                    })
            
            # Prepare chart data from recent rows
            for row in recent_rows:
                try:
                    ear_values.append(float(row.get('EAR', 0)))
                    mar_values.append(float(row.get('MAR', 0)))
                    data.append({
                        'timestamp': row.get('Timestamp', ''),
                        'ear': float(row.get('EAR', 0)),
                        'mar': float(row.get('MAR', 0)),
                        'drowsy': row.get('Drowsy_Alert') == 'True',
                        'yawn': row.get('Yawn_Alert') == 'True',
                        'event': row.get('Event', '')
                    })
                except:
                    continue
        
        # Calculate summary stats
        avg_ear = round(sum(ear_values) / len(ear_values), 3) if ear_values else 0
        avg_mar = round(sum(mar_values) / len(mar_values), 3) if mar_values else 0
        drowsy_pct = round((drowsy_rows / total_rows) * 100, 1) if total_rows > 0 else 0
        yawn_pct = round((yawn_rows / total_rows) * 100, 1) if total_rows > 0 else 0
        
        return jsonify({
            "data": data[-100:],  # Last 100 for charts
            "events": events[-50:],  # Last 50 events
            "summary": {
                "total_records": total_rows,
                "drowsy_percentage": drowsy_pct,
                "yawn_percentage": yawn_pct,
                "avg_ear": avg_ear,
                "avg_mar": avg_mar,
                "total_events": len(events)
            }
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "data": [],
            "summary": {}
        })

if __name__ == '__main__':
    print("🚗 Drowsiness Detection - Simple Frontend")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, threaded=True)
