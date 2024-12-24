import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Function to extract hand landmarks from an image
def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    else:
        return None

# Load dataset
data_dir = 'C:\\Users\\immid\\.spyder-py3\\Indian'  # Replace with the path to your dataset
labels = []
features = []

# Iterate through each subdirectory (each representing a sign)
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        print(f"Processing label: {label}")
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            print(f"Processing image: {image_path}")
            landmarks = extract_hand_landmarks(image_path)
            if landmarks is not None:
                features.append(landmarks)
                labels.append(label)
                print(f"Extracted landmarks for image: {image_file}")
            else:
                print(f"Failed to extract landmarks for image: {image_file}")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Encode labels
label_binarizer = LabelBinarizer()
y_encoded = label_binarizer.fit_transform(y)

# Reshape features for LSTM input (samples, timesteps, features)
# Assuming each sample is a single frame and replicating it for 30 timesteps
X = np.repeat(X[:, np.newaxis, :], 30, axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_binarizer.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Save the model
print("Saving the model...")
model.save('isl_sign_detection_model_lstm.h5')

# Save the label binarizer
with open('label_binarizer.pkl', 'wb') as file:
    pickle.dump(label_binarizer, file)

# Initialize MediaPipe Hands for real-time processing
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Function to extract hand landmarks from a frame
def extract_hand_landmarks_realtime(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    else:
        return None

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    landmarks = extract_hand_landmarks_realtime(frame)
    if landmarks is not None:
        # Reshape landmarks for LSTM input
        input_data = np.repeat(landmarks[np.newaxis, :], 30, axis=0).reshape(1, 30, -1)

        # Predict the sign
        prediction = model.predict(input_data)
        predicted_label = label_binarizer.inverse_transform(prediction)[0]

        # Display the predicted label on the frame
        cv2.putText(frame, f'Sign: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('ISL Sign Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
