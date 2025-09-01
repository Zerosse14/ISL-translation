import base64
import io
import pickle
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# Load the trained model
model_dict = pickle.load(open('./model2.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

labels_dict = {0: 'A', 1: 'Akshay', 2: 'C', 3:'Hello' , 4:"My" , 5:"Name" , 7:"K" ,8:"S" , 9:'H' , 10:'Y'  }

predicted_characters = [0]

app = Flask(__name__)

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISL TS</title>
    <style>
        body {
            background-color: black;
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            display: flex;
            align-items: flex-start;
            background-color: aquamarine;
            padding: 20px;
            border-radius: 10px;
        }
        #video {
            width: 300px;
            height: 300px;
            border-radius: 10px;
            border: 3px solid red;
        }
        .controls {
            display: flex;
            flex-direction: column;
            margin-left: 20px;
        }
        .button {
            margin: 5px 0;
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }
        .output {
            margin-top: 20px;
            font-size: 1.2em;
            color: black;
            background-color: white;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <video id="video" autoplay></video>
        
        <div class="controls">
            <button id="cameraButton" class="button">Open Camera</button>
            <button id="stopCameraButton" class="button" style="display:none;">Close Camera</button>
            
            <div id="predictedCharacter" class="output">Predicted Character: </div>
            <div id="pastCharacters" class="output">Past Characters: </div>
        </div>
    </div>
    
    <script>
    document.addEventListener("DOMContentLoaded", function () {
        const cameraButton = document.getElementById("cameraButton");
        const stopCameraButton = document.getElementById("stopCameraButton");
        const video = document.getElementById("video");
        const predictedCharacterLabel = document.getElementById("predictedCharacter");
        const pastCharactersLabel = document.getElementById("pastCharacters");
        const canvas = document.createElement('canvas');
        let isStreaming = false;
        let lastCharacter = '';

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
                video.srcObject = stream;
                stopCameraButton.style.display = "inline-block";
                cameraButton.style.display = "none";
                isStreaming = true;
                captureFrame();
            } catch (err) {
                console.error("Error accessing camera: ", err);
            }
        }

        function stopCamera() {
            const stream = video.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
            stopCameraButton.style.display = "none";
            cameraButton.style.display = "inline-block";
            isStreaming = false;
        }

        function captureFrame() {
            if (!isStreaming) return;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            const imageData = canvas.toDataURL('image/png');

            fetch('/process_video_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            })
            .then(response => response.json())
            .then(data => {
                if (data && data.predicted_character) {
                    const currentCharacter = data.predicted_character;
                    updateLabel(currentCharacter);
                    if (currentCharacter !== lastCharacter) {
                        lastCharacter = currentCharacter;
                        speakPrediction(currentCharacter);
                    }
                    updatePastCharacters(data.past_characters);
                }
            })
            .catch(error => console.error('Error:', error));

            setTimeout(captureFrame, 1100);
        }

        function updateLabel(character) {
            predictedCharacterLabel.textContent = 'Predicted Character: ' + character;
        }

        function updatePastCharacters(characters) {
            pastCharactersLabel.textContent = 'Past Characters: ' + characters.join(', ');
        }

        function speakPrediction(character) {
            const synth = window.speechSynthesis;
            const utterThis = new SpeechSynthesisUtterance(character);
            synth.speak(utterThis);
        }

        cameraButton.addEventListener("click", startCamera);
        stopCameraButton.addEventListener("click", stopCamera);
    });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_content)

@app.route('/process_video_frame', methods=['POST'])
def process_video_frame():
    data = request.get_json()
    image_data = data['image']
    image_data = image_data.split(',')[1]

    image = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []
    z_ = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z  # Include `z` if necessary
                x_.append(x)
                y_.append(y)
                z_.append(z)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z  # Include `z` if necessary
                data_aux.extend([x - min(x_), y - min(y_)])  # Add z-normalization if needed

    # Flatten the feature vector
    data_flattened = np.ravel(data_aux)

    # Pad the feature vector to match the trained model's input size
    max_size = 84  # Ensure this matches the maximum size used during training
    data_padded = np.pad(data_flattened, (0, max_size - len(data_flattened)), mode='constant')

    # Predict using the trained model
    prediction = model.predict([data_padded])
    predicted_character = labels_dict[int(prediction[0])]

    if predicted_characters[-1] != predicted_character:
        predicted_characters.append(predicted_character)

    return jsonify({
        'predicted_character': predicted_character,
        'past_characters': predicted_characters
    })

if __name__ == '__main__':
    app.run(debug=True)
