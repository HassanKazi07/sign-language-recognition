import cv2
import mediapipe as mp
import os
import csv

# MediaPipe setup
mp_hands = mp.solutions.hands
# We use static_image_mode=True because we are processing disconnected images
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

dataset_path = "dataset"
csv_file = "hand_landmarks.csv"

# Create the header for our CSV file
# It will look like: label, x0, y0, z0, x1, y1, z1... up to z20
header = ['label']
for i in range(21):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

# Find all the folders (A, B, C, del, space, etc.)
classes = sorted([c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))])
print(f"Found {len(classes)} classes.")

# Open the CSV file to start writing
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = [i for i in os.listdir(class_path) if i.endswith(('.jpg', '.jpeg', '.png'))]
        
        # NOTE: Because this uses almost NO RAM, you can use all 3000 images if you want! 
        # But we will stick to your current dataset folder for now.
        print(f"Processing '{class_name}' ({len(images)} images)...")
        
        success_count = 0
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # MediaPipe needs RGB images
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # If a hand is found in the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row = [class_name]
                    
                    # Get the wrist coordinates to use as our base (0,0,0)
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y
                    wrist_z = hand_landmarks.landmark[0].z
                    
                    # Extract all 21 points
                    for lm in hand_landmarks.landmark:
                        # Subtract wrist coordinates to normalize (makes it position-proof!)
                        row.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                    
                    writer.writerow(row)
                    success_count += 1
                    
        print(f"  -> Successfully extracted landmarks for {success_count} hands.")

print(f"\n✅ All done! Your dataset has been converted to {csv_file}")