import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('eye_state_detection_model.h5')
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
closed_eye_counter = 0
alert_threshold = 10
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    print(f"Processing frame {frame_count}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye = roi_gray[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (64, 64))
            eye = eye / 255.0
            eye = eye.reshape(1, 64, 64, 1)
            prediction = model.predict(eye)

            if prediction < 0.5:
                label = 'Closed'
                color = (0, 0, 255)
                closed_eye_counter += 1
            else:
                label = 'Open'
                color = (0, 255, 0)
                closed_eye_counter=0

            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            cv2.putText(frame, label, (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if closed_eye_counter > alert_threshold:
                text = 'Alert'
                cv2.putText(frame, text, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 2)
    cv2.imshow('Eye State Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
