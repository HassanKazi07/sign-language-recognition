import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

dataset_path = "dataset"
csv_file = "hand_landmarks.csv"

header = ['label']
for i in range(21):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

classes = sorted([c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))])
print(f"Found {len(classes)} classes.")

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = [i for i in os.listdir(class_path) if i.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing '{class_name}' ({len(images)} images)...")
        
        success_count = 0
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row = [class_name]
                    
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y
                    wrist_z = hand_landmarks.landmark[0].z
                    
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                    
                    writer.writerow(row)
                    success_count += 1
                    
        print(f"  -> Extracted landmarks for {success_count} hands.")

print(f"\nDataset conversion to {csv_file} complete.")