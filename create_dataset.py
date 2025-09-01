import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from sklearn.utils import resample

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing dataset
DATA_DIR = './data'

# Data and labels
data = []
labels = []
max_size = 84  # Ensure consistency in padding size

def normalize_landmarks(landmarks):
    """Normalize hand landmarks to ensure consistency."""
    x_vals = [lm[0] for lm in landmarks]
    y_vals = [lm[1] for lm in landmarks]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    return [(x - min_x, y - min_y) for x, y in landmarks]

def augment_image(img):
    """Augment image with rotation and flipping."""
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    flipped = cv2.flip(img, 1)  # Horizontal flip
    return [img, rotated, flipped]

def process_image(img):
    """Extract hand landmarks and normalize them."""
    data_aux = []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            normalized_landmarks = normalize_landmarks(landmarks)
            for x, y in normalized_landmarks:
                data_aux.extend([x, y])

    # Pad data to ensure consistent size
    padded_data = np.pad(data_aux, (0, max_size - len(data_aux)), mode='constant')
    return padded_data

# Process all images and labels
for dir_ in os.listdir(DATA_DIR):
    class_data = []  # Temporary list for augmentation
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        augmented_images = augment_image(img)  # Augment images
        
        for aug_img in augmented_images:
            processed_data = process_image(aug_img)
            if len(processed_data) > 0:  # Only add valid frames
                class_data.append(processed_data)
                labels.append(dir_)
    
    # Oversample to balance classes
    if class_data:
        class_data_balanced = resample(
            class_data, replace=True, n_samples=50, random_state=42
        )  # Adjust `n_samples` as needed
        data.extend(class_data_balanced)
        labels.extend([dir_] * len(class_data_balanced))

# Save data to pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data processing complete! Saved {len(data)} samples.")
