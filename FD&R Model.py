import cv2
import numpy as np
import os
from flask import Flask, render_template, Response
import face_recognition

# Initialize Flask app
app = Flask(__name__)

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to encode faces in the dataset
def encode_faces(dataset_path):
    face_encodings = {}
    for person_name in os.listdir(dataset_path): # Iterate through each person folder
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            encodings = []
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img)
                if face_locations:  # Ensure a face is found
                    encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
                    encodings.append(encoding)
            
            if encodings:
                face_encodings[person_name] = np.mean(encodings, axis=0)
    
    return face_encodings

# Encode all faces in the dataset
dataset_path = 'celebrity_dataset'
known_face_encodings = encode_faces(dataset_path)

# Function to recognize faces in the frame
def recognize_face(face_embedding, known_face_encodings, threshold = 0.6):
    min_dist = float('inf')
    identity = None
    for name, encoding in known_face_encodings.items():
        dist = np.linalg.norm(face_embedding - encoding)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > threshold:  # Check if the closest distance is below the threshold
        return "Unknown", min_dist
    return identity, min_dist

# Generator function to stream video frames
def generate_frames():
    cap = cv2.VideoCapture(0) # Start video capture from the default camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frame is captured
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for (x, y, w, h) in faces: # Iterate through detected faces
            face_location = [(y, x + w, y + h, x)]
            face_encodings = face_recognition.face_encodings(rgb_frame, face_location)
            if face_encodings: # If a face encoding is found
                identity, dist = recognize_face(face_encodings[0], known_face_encodings)
                if identity != "Unknown":  # If recognized
                    accuracy = max(0, min(100, 100 - dist * 45))  # Calculate accuracy based on distance
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f'{identity} ({accuracy:.2f}%)',
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (255, 0, 0),
                                2)
                else: # If not recognized
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, 'Unknown', 
                                (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, 
                                (0, 0, 255), 
                                2)

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to be streamed to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask route for the homepage
@app.route('/')
def index():
    return render_template('vediotest.html')

# Flask route for streaming video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
