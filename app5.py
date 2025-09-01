import base64
import io
import pickle
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C'}

app = Flask(__name__)

# HTML and JS content combined into the Flask Python file
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISL TS</title>
    <style>
        body{
            background-color: black;
            height: 100vh;
        }
        .navbar{
            height: 70px;
            background-color: red;
            color:white;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .content{
            background-color: aquamarine;
            width: 1000px;
            height:500px;
            margin-top: 40px;
            margin-left: 250px;
            margin-right: 250px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .c1{
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        #cameraButton {
            padding: 10px 20px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        #cameraContainer {
            display: none;
        }
        #predictedCharacter {
            margin-top: 20px;
            font-size: 2em;
            color: white;
            background-color: black;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>INDIAN SIGN LANGUAGE TRANSLATION SYSTEM</h1>
    </div>
    <div class="content">
        <div class="c1"> 
            <button id="cameraButton">Open Camera</button>
            <div id="cameraContainer">
                <video id="video" width="590" height="590" autoplay></video>
            </div>
            <div id="predictedCharacter">Predicted Character: </div> <!-- Label to show predicted character -->
        </div>
    </div>
    
    <script>
document.addEventListener("DOMContentLoaded", function () {
    const cameraButton = document.getElementById("cameraButton");
    const cameraContainer = document.getElementById("cameraContainer");
    const video = document.getElementById("video");
    const predictedCharacterLabel = document.getElementById("predictedCharacter");
    const canvas = document.createElement('canvas'); // Create a canvas element to capture video frames
    let isStreaming = false;
    let lastCharacter = ''; // Track the last predicted character

    // Function to start camera
    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
            video.srcObject = stream;
            cameraContainer.style.display = "block";
            isStreaming = true;
            captureFrame(); // Start capturing frames after camera starts
        } catch (err) {
            console.error("Error accessing camera: ", err);
        }
    }

    // Capture a video frame and send it to the server
    function captureFrame() {
        if (!isStreaming) return;

        // Draw the current video frame on the canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert the canvas image to Base64
        const imageData = canvas.toDataURL('image/png');

        // Send the frame to the Flask server
        fetch('/process_video_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Response from server:', data);  // Debug: Check if you're getting a response
            if (data && data.predicted_character) {
                const currentCharacter = data.predicted_character;
                
                // Update the label
                updateLabel(currentCharacter);

                // Check if the character has changed, then provide voice feedback
                if (currentCharacter !== lastCharacter) {
                    lastCharacter = currentCharacter; // Update the last character
                    speakPrediction(currentCharacter); // Voice feedback
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });

        // Capture another frame after a delay (e.g., 100ms)
        setTimeout(captureFrame, 100);
    }

    // Update the predicted character label
    function updateLabel(character) {
        predictedCharacterLabel.textContent = 'Predicted Character: ' + character;
    }

    // Speak the predicted character using the Web Speech API
    function speakPrediction(character) {
        const synth = window.speechSynthesis;
        const utterThis = new SpeechSynthesisUtterance(character);
        synth.speak(utterThis);
    }

    // Event listener for camera button click
    cameraButton.addEventListener("click", function () {
        startCamera();
    });
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

    # Remove the prefix 'data:image/png;base64,' from the image string
    image_data = image_data.split(',')[1]
    
    # Decode the Base64 image
    image = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image))

    # Convert the image to OpenCV format
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Debug: Print confirmation that a frame was received
    print("Received a frame")

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
        print(f"Predicted Character: {predicted_character}")
    else:
        predicted_character = "No hand detected"
        print("No hand detected")

    return jsonify({'predicted_character': predicted_character})

if __name__ == '__main__':
    app.run(debug=True)