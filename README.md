Sign Language Recognition System
A high-accuracy, real-time sign language translation interface built with Python, MediaPipe, and TensorFlow. This project was developed for a final year diploma project.

🚀 Key Features
99.23% Accuracy: Achieved using a custom Feed-Forward Neural Network trained on geometric hand landmarks.

MediaPipe Integration: Real-time 21-point hand skeleton tracking for robust detection.

Ambidextrous Support: Software-level mirroring allows the system to recognize signs from both left and right hands.

Live Sentence Formation: Includes a "radar-style" top-to-bottom scanner that captures characters and builds sentences in a web interface.

🛠️ Tech Stack
Backend: Flask (Python)

AI/ML: TensorFlow, Keras, MediaPipe, Scikit-learn

Data Processing: Pandas, NumPy

Frontend: HTML5, CSS3 (Modern Responsive UI)

📋 Installation & Setup
To run this project locally, ensure you have Python 3.10 installed.

Clone the repository:

Bash
git clone https://github.com/YOUR_USERNAME/sign-language-recognition.git
cd sign-language-recognition
Install dependencies:
Using the provided requirements.txt to ensure version compatibility (specifically NumPy 1.26.4):

Bash
pip install -r requirements.txt
Run the Application:

Bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000.

📖 How It Works
The Scanner Line moves from top to bottom every 1.33 seconds.

When the line hits the bottom, the system extracts the 63 (x,y,z) coordinates of your hand landmarks.

The model predicts the character and adds it to the Live Formed Sentence box.

Use the 'del' sign to remove a character or the 'space' sign to add a space.