from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load the lightweight model and the label encoder
model = tf.keras.models.load_model("model/sign_model_csv.h5", compile=False)

with open('model/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

# Prediction Throttling & UI
frame_count = 0
PREDICT_EVERY_N_FRAMES = 40 
last_prediction = "No Hand"
current_sentence = ""

def generate_frames():
    global frame_count, last_prediction, current_sentence
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        frame_count += 1
        h, w, c = frame.shape
        
        # --- UPDATED UI: Top-to-Bottom Scanner Line ---
        # Calculate how far down the screen the line should be (0.0 to 1.0)
        progress = (frame_count % PREDICT_EVERY_N_FRAMES) / PREDICT_EVERY_N_FRAMES
        sweep_y = int(progress * h)
        
        # Draw a clean, crisp horizontal line moving downwards (Green, 2px thick)
        cv2.line(frame, (0, sweep_y), (w, sweep_y), (0, 255, 0), 2)

        current_prediction = last_prediction
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # The Ambidextrous Hack
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                wrist_z = hand_landmarks.landmark[0].z
                
                row = []
                mirrored_row = []
                
                for lm in hand_landmarks.landmark:
                    nx = lm.x - wrist_x
                    ny = lm.y - wrist_y
                    nz = lm.z - wrist_z
                    
                    row.extend([nx, ny, nz])
                    mirrored_row.extend([-nx, ny, nz]) 
                
                # Only predict when the scanner hits the bottom of the screen
                if frame_count % PREDICT_EVERY_N_FRAMES == 0:
                    
                    # Flash the screen border green to show a scan happened!
                    cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 10)
                    
                    input_data = np.array([row], dtype=np.float32)
                    mirrored_data = np.array([mirrored_row], dtype=np.float32)
                    
                    # Predict on BOTH the normal hand and the mirrored hand
                    pred_normal = model.predict(input_data, verbose=0)
                    pred_mirrored = model.predict(mirrored_data, verbose=0)
                    
                    conf_normal = np.max(pred_normal) * 100
                    conf_mirrored = np.max(pred_mirrored) * 100
                    
                    # Trust whichever prediction is more confident
                    if conf_normal >= conf_mirrored:
                        best_pred = pred_normal
                        confidence = conf_normal
                    else:
                        best_pred = pred_mirrored
                        confidence = conf_mirrored
                        
                    index = np.argmax(best_pred)
                    predicted_char = encoder.inverse_transform([index])[0]
                    
                    last_prediction = f"{predicted_char} ({confidence:.0f}%)"
                    current_prediction = last_prediction
                    
                    if confidence > 85: 
                        if predicted_char == 'space':
                            current_sentence += " "
                        elif predicted_char == 'del':
                            current_sentence = current_sentence[:-1]
                        else:
                            current_sentence += predicted_char
                            
        cv2.putText(frame, current_prediction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence')
def get_sentence():
    return jsonify({"sentence": current_sentence})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global current_sentence
    current_sentence = ""
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)