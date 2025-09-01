import os
import cv2
import time

# Directory to save the data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 50
dataset_size = 50

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Loop through each class
for j in range(27,50):
    # Create directory for the current class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for the user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing the camera. Please check your setup.")
            break
        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Wait for 3 seconds before starting the capture
    print("Starting capture in 3 seconds...")
    time.sleep(3)

    # Capture images for the dataset
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing the camera. Please check your setup.")
            break
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1
        cv2.waitKey(25)

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
