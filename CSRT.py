import pigpio
from time import sleep
import cv2
from picamera2 import Picamera2

pi = pigpio.pi()
SERVO_PIN = 14
CENTER_PW = 1500
MIN_PW = 500
MAX_PW = 2500
pi.set_servo_pulsewidth(SERVO_PIN, CENTER_PW)
last_pw = CENTER_PW

picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
sleep(2)

face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")

frame_width = 320
DEAD_ZONE_NORM = 0.06
alpha = 0.5
MAX_STEP = 80
MIN_DELTA_PW = 5
HISTORY_LEN = 5

tracker = cv2.TrackerCSRT_create()
tracking = False
pos_history = []

while True:
    frame = picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if not tracking:
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))
            tracking = True

    if tracking:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            center_x = x + w // 2
            normalized_pos = (center_x / frame_width) * 2 - 1
            normalized_pos = max(min(normalized_pos, 1), -1)

            if abs(normalized_pos) < DEAD_ZONE_NORM:
                adjusted_pos = 0
            else:
                adjusted_pos = normalized_pos

            pos_history.append(adjusted_pos)
            if len(pos_history) > HISTORY_LEN:
                pos_history.pop(0)
            smoothed_pos = sum(pos_history) / len(pos_history)

            offset = -smoothed_pos * 1000
            target_pw = CENTER_PW + offset
            target_pw = max(min(target_pw, MAX_PW), MIN_PW)

            diff = target_pw - last_pw

            if abs(diff) < MIN_DELTA_PW:
                new_pw = last_pw
            else:
                diff = max(min(diff, MAX_STEP), -MAX_STEP)
                new_pw = last_pw + alpha * diff
                pi.set_servo_pulsewidth(SERVO_PIN, new_pw)
                last_pw = new_pw

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, y + h // 2), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Pulse: {int(new_pw)}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            tracking = False
            pos_history.clear()

    cv2.imshow("Tracking Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pi.set_servo_pulsewidth(SERVO_PIN, 0)
cv2.destroyAllWindows()
