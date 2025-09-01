import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time  # Import the time module to control the speech delay

# Load the model and labels
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label dictionary for predicted letters
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

# Initialize pyttsx3 for text-to-speech conversion
engine = pyttsx3.init()

# Adjust voice properties (optional)
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Variable to store the previously predicted character and the time of the last spoken letter
previous_character = None
last_speech_time = 0  # Tracks when the last letter was spoken
speech_delay = 2  # Minimum delay in seconds between speaking letters

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make a prediction based on the hand landmark data
        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        # Draw the bounding box and display the predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        # Check if the predicted character has changed and if the delay has passed before converting it to speech
        current_time = time.time()
        if predicted_character != previous_character and (current_time - last_speech_time) > speech_delay:
            previous_character = predicted_character  # Update the last spoken character
            last_speech_time = current_time  # Update the last speech time

            # Convert the predicted letter to speech
            engine.say(predicted_character)
            engine.runAndWait()  # This makes sure speech happens immediately

    # Display the frame with the prediction
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
