import cv2
import math
import numpy as np
from detect_one_car import VehicleFollower
from estimate_speed import estimate_speed
from homography_map import transform_point

# --- Configuration ---
VIDEO_PATH = "traffic.mp4"
FPS = 30
METERS_PER_PIXEL = 0.088766
SPEED_LIMIT_KMPH = 30
RESIZE_SCALE = 0.3

# --- Load Homography ---
H = np.load("homography_matrix.npy")
map_width, map_height = np.load("map_size.npy")

# --- Colors ---
COLOR_NORMAL = (0, 255, 0)
COLOR_ALERT = (0, 0, 255)

# --- Init ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

follower = VehicleFollower()

# Read first frame for manual vehicle selection
ret, frame = cap.read()
if not ret:
    print("Error reading first frame.")
    cap.release()
    exit()

frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
follower.select_vehicle(frame)

previous_center = None
map_img = np.zeros((map_height, map_width, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    result = follower.detect_and_follow(frame)

    if result is not None:
        (x1, y1, x2, y2), center = result
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cx, cy = map(int, center)

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NORMAL, 2)
        cv2.circle(frame, (cx, cy), 4, COLOR_NORMAL, -1)

        if previous_center is not None:
            dx = cx - previous_center[0]
            dy = cy - previous_center[1]
            pixels_moved = math.hypot(dx, dy)

            if pixels_moved > 1:
                speed_kmph = estimate_speed(pixels_moved, FPS, METERS_PER_PIXEL)
                color = COLOR_ALERT if speed_kmph > SPEED_LIMIT_KMPH else COLOR_NORMAL
                cv2.putText(frame, f"{speed_kmph:.2f} km/h", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        previous_center = (cx, cy)

        map_pos = transform_point(H, (cx, cy))
        cv2.circle(map_img, map_pos, 4, (255, 255, 255), -1)

    cv2.imshow("Vehicle Speed Estimation", frame)
    cv2.imshow("Top-Down Map", map_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()