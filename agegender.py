import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('age_gender_model.h5', compile=False)
print("Model loaded successfully.")

# Gender mapping
gender_dict = {0: 'Male', 1: 'Female'}

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for better compatibility on Windows

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face = gray_frame[y:y+h, x:x+w]

        # Convert grayscale to RGB (if needed)
        face_resized = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

        # Preprocess the face
        face_resized = cv2.resize(face_resized, (128, 128))  # Resize to match model input size
        face_array = img_to_array(face_resized) / 255.0  # Normalize
        face_array = np.expand_dims(face_array, axis=0)  # Expand dimensions to match model input

        # Predict age and gender
        predictions = model.predict(face_array)
        gender_pred = predictions[0]  # First output (gender)
        age_pred = predictions[1]  # Second output (age)

        # Extract gender and age
        gender_label = gender_dict[np.argmax(gender_pred)]
        predicted_age = round(age_pred[0][0])

        # Display the results
        label = f"{gender_label}, Age: {predicted_age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Age and Gender Prediction', frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
