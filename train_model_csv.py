import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

csv_file = "hand_landmarks.csv"

print("Loading dataset...")
# 1. Load the CSV data
df = pd.read_csv(csv_file)

# 2. Separate labels (the letter/sign) and features (the 63 coordinates)
X = df.drop('label', axis=1).values
y_text = df['label'].values

# 3. Convert text labels ('A', 'B', 'space') into numbers (0, 1, 27)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_text)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

num_classes = len(encoder.classes_)
print(f"Found {num_classes} classes: {encoder.classes_}")

# 4. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_test)}")

# 5. Build a lightweight Neural Network for coordinates
model = models.Sequential([
    layers.InputLayer(input_shape=(63,)), # 21 landmarks * 3 coordinates (x,y,z)
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train the model
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50, # We can run more epochs because it's so fast!
    batch_size=32
)

# 7. Save the model and the label encoder
model.save("model/sign_model_csv.h5")

# Save the encoder so app.py knows which number means which letter
with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("\n✅ New high-accuracy model trained and saved successfully!")