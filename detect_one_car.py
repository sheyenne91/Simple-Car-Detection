import torch
import cv2
import numpy as np

class VehicleFollower:
    def __init__(self, model_path='yolov5s.pt', classes_to_detect=[2, 3, 5, 7]):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.classes_to_detect = classes_to_detect
        self.previous_center = None
        self.selected_box = None

    def select_vehicle(self, frame):
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()

        vehicle_boxes = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) in self.classes_to_detect:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                vehicle_boxes.append(((x1, y1, x2, y2), (cx, cy)))

        if not vehicle_boxes:
            print("No vehicles detected.")
            return

        clone = frame.copy()
        for (x1, y1, x2, y2), _ in vehicle_boxes:
            cv2.rectangle(clone, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

        selected_point = []

        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(selected_point) == 0:
                selected_point.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)

        cv2.namedWindow("Click a vehicle to track")
        cv2.setMouseCallback("Click a vehicle to track", click)

        while len(selected_point) < 1:
            cv2.imshow("Click a vehicle to track", clone)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow("Click a vehicle to track")
        click_x, click_y = selected_point[0]
        self.selected_box = min(vehicle_boxes, key=lambda b: np.linalg.norm(np.array(b[1]) - (click_x, click_y)))
        self.previous_center = self.selected_box[1]

    def detect_and_follow(self, frame):
        if self.selected_box is None:
            return None

        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()

        vehicle_boxes = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) in self.classes_to_detect:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                vehicle_boxes.append(((x1, y1, x2, y2), (cx, cy)))

        if not vehicle_boxes:
            return None

        selected = min(vehicle_boxes, key=lambda b: np.linalg.norm(np.array(b[1]) - self.previous_center))
        self.previous_center = selected[1]
        return selected[0], selected[1]