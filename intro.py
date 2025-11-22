import cv2
from deepface import DeepFace
import os
db_base = 'database/'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
print('press Q to quit!')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        cv2.putText(frame, "No faces", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    for i, (x, y, w, h) in enumerate(faces):
        pad = int(0.15 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        face_img = frame[y1:y2, x1:x2]
        name_label = 'Unknown'
        emotion_label = 'N/A'
        try:
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            emotion_label = analysis.get('dominant_emotion', emotion_label)
        except Exception:
            pass
        try:
            results = DeepFace.find(img_path=face_img, db_path=db_base, enforce_detection=False)
            if isinstance(results, list) and len(results) > 0 and len(results[0]) > 0:
                identity = results[0]['identity'][0]
                name_label = os.path.basename(identity).split('.')[0]
        except Exception:
            pass
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name_label} | {emotion_label}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()