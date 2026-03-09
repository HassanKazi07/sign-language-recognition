Sign Language AI - Neural Recognition MVP v2.0
An AI-powered real-time Sign Language to Speech translation system. This project utilizes a multi-layered AI pipeline to bridge the communication gap for the Deaf and Hard of Hearing community.

🚀 Key Features in v2.0
Neural Hand Tracking: Real-time 3D landmark extraction using MediaPipe.

Deep Learning Classifier: Custom Keras model trained on 63 coordinate features (21 landmarks × 3 axes).

NLP Autocorrect: Integrated spell-checking logic to handle human signing jitter.

Threaded Text-to-Speech: High-fidelity Google TTS voice that speaks words/sentences without lagging the video feed.

Premium UI: Dark-mode dashboard with a zig-zag landing page and live audio waveform visualizer.

🧠 The AI Architecture
The system operates on three distinct levels of Artificial Intelligence:

Computer Vision (MediaPipe): Processes the raw webcam feed to identify 21 hand joints. It normalizes these coordinates relative to the wrist to ensure the model is position-proof.

Deep Learning (Keras/TensorFlow): A Sequential Neural Network evaluates the extracted landmarks and predicts the signed character based on statistical probability.

NLP & TTS (SpellChecker & gTTS): * Single Space: Triggers an autocorrect check on the last word and speaks it aloud.

Double Space: Triggers a full sentence translation and playback.


Shutterstock
Explore
🛠️ Installation & Setup
Clone the repository:

Bash
git clone <https://github.com/HassanKazi07/sign-language-recognition>
Install dependencies:

Bash
pip install -r requirements.txt
Run the application:

Bash
python app.py
Access the UI:
Open your browser and go to http://127.0.0.1:5000

👥 Developed By
Hassan Kazi • Ayaan Shaikh • Namir Mapari • Ashfaque Chhatbhar
