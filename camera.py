import cv2

# Find available cameras
available_cameras = []
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()

print(f"Available cameras: {available_cameras}")

if available_cameras:
    # Use the first available camera
    cap = cv2.VideoCapture(available_cameras[0])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        cv2.imshow('Camera Test', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
else:
    print("No cameras found!")