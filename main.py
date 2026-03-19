import cv2
import math
import os
import time

counted_ids = set()

if not os.path.exists("faces"):
    os.makedirs("faces")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

face_centers = {}
face_sides = {}
face_id_count = 0
last_count_time = {}

entry_count = 0
exit_count = 0
line_y = 250

def get_center(x, y, w, h):
    return (x + w//2, y + h//2)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=8,
    minSize=(80, 80)
    )
    new_centers = {}
    new_sides = {}
    cv2.line(frame, (0, line_y), (640, line_y), (0,255,255), 2)

    counted_ids = set()
    for (x, y, w, h) in faces:
        if w < 80 or h < 80:
            continue
        if x < 100 or x > 540:
            continue

        center = get_center(x, y, w, h)
        same_object_detected = False
        current_id = None

        for id, prev_center in face_centers.items():
            if distance(center, prev_center) < 500:
                new_centers[id] = center
                same_object_detected = True
                current_id = id

                prev_side = face_sides.get(id, "above" if prev_center[1] < line_y else "below")
                curr_side = "above" if center[1] < line_y else "below"
                new_sides[id] = curr_side

                if id not in counted_ids:
                    if prev_side == "above" and curr_side == "below":
                        if id not in counted_ids:
                            entry_count += 1
                            counted_ids.add(id)
                        counted_ids.add(id)
                    elif prev_side == "below" and curr_side == "above":
                        exit_count += 1
                        counted_ids.add(id)

                cv2.putText(frame, f'ID {id}', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                break

   
        if not same_object_detected:
            face_id_count += 1
            new_centers[face_id_count] = center
            current_id = face_id_count
            new_sides[face_id_count] = "above" if center[1] < line_y else "below"

            cv2.putText(frame, f'ID {face_id_count}', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(f"faces/face_{current_id}_{x}.jpg", face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

    face_centers = new_centers
    face_sides = new_sides

    cv2.putText(frame, f'Total: {len(face_centers)}', (10,30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f'Entry: {entry_count}', (10,60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(frame, f'Exit: {exit_count}', (10,90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("VisiTrack AI - Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()