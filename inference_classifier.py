import base64
import io
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import pickle
from flask import Flask, render_template, request, jsonify

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels for prediction
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data['image']
    
    # Remove the prefix 'data:image/png;base64,' from the image string
    image_data = image_data.split(',')[1]
    
    # Decode the Base64 image
    image = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image))

    # Convert the image to OpenCV format
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Process the frame using Mediapipe Hands
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
    else:
        predicted_character = "No hand detected"

    return jsonify({'predicted_character': predicted_character})

if __name__ == '__main__':
    app.run(debug=True)
