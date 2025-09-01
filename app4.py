import sys
import cv2
import pickle
import numpy as np
import base64
import io
from PIL import Image
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
import mediapipe as mp

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

labels_dict = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
    11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
    21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z',
    27: 'hello', 28: 'Indian', 29: 'language', 30: 'again', 31: 'me / I',
    32: 'you', 33: 'man', 34: 'woman', 35: 'deaf', 36: 'hearing',
    37: 'teacher', 38: 'thank you', 39: 'welcome', 40: 'please',
    41: 'sorry', 42: 'namaste', 43: 'how are you', 44: 'I’m fine',
    45: 'good', 46: 'bad', 47: 'wrong', 48: 'strong', 49: 'understand'
}
predicted_characters = [0]

class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ISL Translator")
        self.setGeometry(100, 100, 800, 600)

        # GUI Elements
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black;")

        self.predicted_label = QLabel("Predicted Character: ")
        self.past_characters_label = QLabel("Past Characters: ")

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.predicted_label)
        layout.addWidget(self.past_characters_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

        # Video Capture Variables
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_camera(self):
        self.timer.stop()
        self.capture.release()
        self.video_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame_rgb, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(image))

            # Process frame for hand recognition
            prediction, past = self.process_frame(frame)
            self.predicted_label.setText(f"Predicted Character: {prediction}")
            self.past_characters_label.setText(f"Past Characters: {', '.join(past)}")

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_ = []
        y_ = []

        predicted_character = "None"

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

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            if predicted_characters[-1] != predicted_character:
                predicted_characters.append(predicted_character)

        return predicted_character, predicted_characters

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
