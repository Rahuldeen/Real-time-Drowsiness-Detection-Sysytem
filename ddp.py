import cv2
import mediapipe as mp
from scipy.spatial import distance
import pygame
import numpy as np
import pyttsx3
import threading
import time
import csv
from datetime import datetime

# Voice alert state
voice_alert_lock = threading.Lock()
last_voice_time = 0

def speak_alert(message="Please stay awake!"):
    """Speak voice alert in background thread"""
    global last_voice_time
    
    with voice_alert_lock:
        current_time = time.time()
        if current_time - last_voice_time < 3:  # Prevent spam (3 sec cooldown)
            return
        last_voice_time = current_time
    
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

def eye_aspect_ratio(eye_points):
    """Calculate Eye Aspect Ratio"""
    # Vertical distances
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    # Horizontal distance
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_points):
    """Calculate Mouth Aspect Ratio (MAR)
    
    Similar to EAR, MAR measures the openness of the mouth.
    Higher MAR = mouth more open (yawning)
    Lower MAR = mouth closed
    
    mouth_points should contain 8 landmarks:
    [0] - left corner, [1] - right corner (horizontal)
    [2,3] - upper lip vertical points
    [4,5] - lower lip vertical points  
    [6,7] - additional vertical points for accuracy
    """
    # Vertical distances (lip opening)
    A = distance.euclidean(mouth_points[2], mouth_points[6])  # Upper to lower lip (left side)
    B = distance.euclidean(mouth_points[3], mouth_points[7])  # Upper to lower lip (right side)
    C = distance.euclidean(mouth_points[4], mouth_points[5])  # Upper to lower lip (center)
    
    # Horizontal distance (mouth width)
    D = distance.euclidean(mouth_points[0], mouth_points[1])
    
    # MAR formula: average of vertical distances / horizontal distance
    mar = (A + B + C) / (3.0 * D)
    return mar

def main():
    # Initialize components
    print("Starting Camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("!!! Could not access camera. Exiting...")
        return
    
    print("Loading MediaPipe Face Mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("Initializing alert system...")
    pygame.mixer.init()
    
    # MediaPipe face landmarks for eyes
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # MediaPipe face landmarks for mouth (outer and inner lip contour)
    # [left corner, right corner, upper lip top, upper lip bottom, 
    #  lower lip top, lower lip bottom, inner upper, inner lower]
    MOUTH_INDICES = [61, 291, 39, 0, 269, 405, 13, 14]
    
    # Parameters
    EAR_THRESHOLD = 0.25
    MAR_THRESHOLD = 0.6  # Threshold for yawn detection
    FRAME_CHECK = 60  # ~2 seconds at 30fps
    flag = 0
    yawn_flag = 0
    alert_active = False
    yawn_alert_active = False
    
    print("Starting drowsiness detection. Press ESC to exit.")
    
    # Setup CSV logging - single file, append mode
    csv_filename = "detection_log.csv"
    import os
    file_exists = os.path.exists(csv_filename)
    csv_file = open(csv_filename, 'a', newline='')  # Append mode
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(['Timestamp', 'EAR', 'MAR', 'Face_Detected', 'Drowsy_Alert', 'Yawn_Alert', 'Event'])
    print(f"Logging data to: {csv_filename} (append mode)")
    
    # Logging control
    frame_count = 0
    LOG_INTERVAL = 30  # Log every 30 frames (~1 per second)
    prev_alert_active = False
    prev_yawn_alert = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        ear = 0
        mar = 0
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Get image dimensions
            h, w = frame.shape[:2]
            
            # Extract eye coordinates
            left_eye_points = []
            right_eye_points = []
            
            for idx in LEFT_EYE_INDICES:
                landmark = landmarks[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                left_eye_points.append((x, y))
                
            for idx in RIGHT_EYE_INDICES:
                landmark = landmarks[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                right_eye_points.append((x, y))
            
            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            ear = (left_ear + right_ear) / 2.0
            
            # Extract mouth coordinates
            mouth_points = []
            for idx in MOUTH_INDICES:
                landmark = landmarks[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                mouth_points.append((x, y))
            
            # Calculate MAR
            mar = mouth_aspect_ratio(mouth_points)
            
            # Draw eye landmarks
            for point in left_eye_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
            for point in right_eye_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
            
            # Draw mouth landmarks
            for point in mouth_points:
                cv2.circle(frame, point, 2, (255, 0, 255), -1)
        
        # Display information
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        status_text = "Face detected" if face_detected else "No face detected"
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Drowsiness detection logic
        if ear < EAR_THRESHOLD and face_detected:
            flag += 1
            cv2.putText(frame, f"Eyes closed: {flag}/{FRAME_CHECK}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if flag >= FRAME_CHECK and not alert_active:
                print("!!! DROWSINESS DETECTED! Playing alert...")
                # Voice alert
                speak_alert("Please stay awake!")
                # Try to play sound file, else use system beep
                try:
                    pygame.mixer.music.load("alert.wav")
                    pygame.mixer.music.play()
                except:
                    try:
                        import winsound
                        winsound.Beep(1000, 1000)  # Frequency, duration
                    except:
                        print("BEEP! DROWSINESS ALERT!")
                
                alert_active = True
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            flag = 0
            alert_active = False
        
        # Smart CSV logging: every N frames OR on alert state changes
        frame_count += 1
        should_log = False
        event = "periodic"
        
        # Log on alert state changes (immediate)
        if alert_active != prev_alert_active:
            should_log = True
            event = "DROWSY_START" if alert_active else "DROWSY_END"
            prev_alert_active = alert_active
        
        if yawn_alert_active != prev_yawn_alert:
            should_log = True
            event = "YAWN_START" if yawn_alert_active else "YAWN_END"
            prev_yawn_alert = yawn_alert_active
        
        # Log every N frames for trend analysis
        if frame_count >= LOG_INTERVAL:
            should_log = True
            frame_count = 0
        
        if should_log:
            csv_writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                round(ear, 4),
                round(mar, 4),
                face_detected,
                alert_active,
                yawn_alert_active,
                event
            ])
            csv_file.flush()  # Save to disk immediately
        
        cv2.imshow("Drowsiness Detection - Press ESC to exit", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            print("Exiting program...")
            break
    
    # Cleanup
    csv_file.close()
    print(f"Data saved to: {csv_filename}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()