
import cv2
import tensorflow as tf
import numpy as np

# Load pre-trained gesture recognition model (optimized for TensorFlow Lite on Jetson)
interpreter = tf.lite.Interpreter(model_path="gesture_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to perform gesture recognition
def recognize_gesture(frame):
    # Preprocess frame for model input
    img_resized = cv2.resize(frame, (224, 224))
    img_normalized = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_normalized)
    interpreter.invoke()

    # Get output and decode gesture
    output_data = interpreter.get_tensor(output_details[0]['index'])
    gesture_index = np.argmax(output_data)
    gestures = ["wave", "point", "thumbs_up", "none"]
    gesture = gestures[gesture_index]
    return gesture

# Start capturing video
cap = cv2.VideoCapture(0)  # Adjust device index for external cameras
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gesture = recognize_gesture(frame)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
