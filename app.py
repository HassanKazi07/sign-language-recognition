from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import time
import threading
import io
from spellchecker import SpellChecker 
from gtts import gTTS
import pygame

app = Flask(__name__)

pygame.mixer.init()

spell = SpellChecker() 
spell.word_frequency.load_words(['hassan'])

model = tf.keras.models.load_model("model/sign_model_csv.h5", compile=False)

with open('model/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

frame_count = 0
PREDICT_EVERY_N_FRAMES = 40 
last_prediction = "No Hand"
current_sentence = ""
last_space_time = 0.0  

# NEW: Global flag to track if the AI is currently talking
is_speaking = False

def speak_text(text):
    global is_speaking
    if not text or not text.strip():
        return
    try:
        is_speaking = True # Turn on the robot's mouth
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        
        # Keep the flag True while the audio is actively playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        print(f"Audio Engine Error: {e}")
    finally:
        is_speaking = False # Turn off the robot's mouth

def generate_frames():
    global frame_count, last_prediction, current_sentence, last_space_time
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        frame_count += 1
        h, w, c = frame.shape
        
        progress = (frame_count % PREDICT_EVERY_N_FRAMES) / PREDICT_EVERY_N_FRAMES
        sweep_y = int(progress * h)
        cv2.line(frame, (0, sweep_y), (w, sweep_y), (36, 112, 242), 2)

        current_prediction = last_prediction
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                custom_dot = mp_draw.DrawingSpec(color=(36, 112, 242), thickness=4, circle_radius=2)
                custom_line = mp_draw.DrawingSpec(color=(68, 31, 10), thickness=2)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, custom_dot, custom_line)
                
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
                
                if frame_count % PREDICT_EVERY_N_FRAMES == 0:
                    
                    cv2.rectangle(frame, (0,0), (w,h), (36, 112, 242), 10)
                    
                    input_data = np.array([row], dtype=np.float32)
                    mirrored_data = np.array([mirrored_row], dtype=np.float32)
                    
                    pred_normal = model(input_data, training=False).numpy()
                    pred_mirrored = model(mirrored_data, training=False).numpy()
                    
                    conf_normal = np.max(pred_normal) * 100
                    conf_mirrored = np.max(pred_mirrored) * 100
                    
                    if conf_normal >= conf_mirrored:
                        best_pred = pred_normal
                        confidence = conf_normal
                    else:
                        best_pred = pred_mirrored
                        confidence = conf_mirrored
                        
                    index = np.argmax(best_pred)
                    predicted_char = encoder.inverse_transform([index])[0].lower()
                    
                    last_prediction = f"{predicted_char.upper()} ({confidence:.0f}%)"
                    current_prediction = last_prediction
                    
                    if confidence > 85: 
                        if predicted_char == 'space':
                            current_time = time.time()
                            
                            if current_time - last_space_time < 2.0:
                                threading.Thread(target=speak_text, args=(current_sentence,)).start()
                                last_space_time = 0.0 
                            else:
                                words = current_sentence.split()
                                if len(words) > 0:
                                    last_word = words[-1]
                                    corrected_word = spell.correction(last_word)
                                    
                                    if corrected_word is not None:
                                        words[-1] = corrected_word
                                        
                                    current_sentence = " ".join(words) + " "
                                    word_to_speak = corrected_word if corrected_word is not None else last_word
                                    threading.Thread(target=speak_text, args=(word_to_speak,)).start()
                                else:
                                    current_sentence += " "
                                
                                last_space_time = current_time
                                
                        elif predicted_char == 'del':
                            current_sentence = current_sentence[:-1]
                        else:
                            current_sentence += predicted_char
                            
        cv2.putText(frame, current_prediction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (68, 31, 10), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translator')
def translator():
    return render_template('translator.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence')
def get_sentence():
    # NEW: Now we also send the 'is_speaking' status to the web browser!
    return jsonify({
        "sentence": current_sentence.capitalize(),
        "is_speaking": is_speaking 
    })

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global current_sentence, last_space_time
    current_sentence = ""
    last_space_time = 0.0
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)