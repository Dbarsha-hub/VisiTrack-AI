import cv2
import math
import os

if not os.path.exists("faces"):
    os.makedirs("faces")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

face_centers = {}
face_id_count = 0

def get_center(x, y, w, h):
    return (x + w//2, y + h//2)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    new_centers = {}

   
    for (x, y, w, h) in faces:

        center = get_center(x, y, w, h)
        same_object_detected = False

        for id, prev_center in face_centers.items():
            if distance(center, prev_center) < 100:
                new_centers[id] = center
                same_object_detected = True

                cv2.putText(frame, f'ID {id}', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                break

        if not same_object_detected:
            face_id_count += 1
            new_centers[face_id_count] = center

            cv2.putText(frame, f'ID {face_id_count}', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

       
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(f"faces/face_{face_id_count}_{x}.jpg", face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

    
    for id in new_centers:
        face_centers[id] = new_centers[id]

    cv2.putText(frame, f'Total: {len(face_centers)}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("VisiTrack AI - Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()