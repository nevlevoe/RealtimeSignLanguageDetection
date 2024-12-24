import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle

# Load the trained model and label binarizer
model = load_model('isl_sign_detection_model_lstm_enlarged.h5')
with open('label_binarizer.pkl', 'rb') as file:
    label_binarizer = pickle.load(file)

# Initialize MediaPipe Hands for real-time processing
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

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
        print("Failed to capture image.")
        break

    landmarks = extract_hand_landmarks_realtime(frame)
    if landmarks is not None:
        input_data = np.repeat(landmarks[np.newaxis, :], 30, axis=0).reshape(1, 30, -1)
        prediction = model.predict(input_data)
        predicted_label = label_binarizer.inverse_transform(prediction)[0]
        cv2.putText(frame, f'Sign: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('ISL Sign Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
