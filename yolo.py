import torch
import cv2
import numpy as np
from database import log_detection
from collections import defaultdict
from ultralytics import YOLO

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects


class YOLOv8Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # تحميل نموذج YOLOv8
        self.classes = self.model.names  # أسماء التصنيفات المدعومة من النموذج
        self.tracker = CentroidTracker()
        self.detected_boxes = []
        self.logged_objects = set()  # Set to keep track of logged object IDs

    def detect(self, frame):
        results = self.model(frame)  # الكشف عن الكائنات
        detections = results[0].boxes if results and len(results) > 0 else []

        self.detected_boxes = []

        rects = []
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.classes[class_id]

            # التحقق مما إذا كانت التصنيفات المكتشفة هي "person" أو "backpack" أو "handbag"
            if class_name in ['person', 'backpack', 'handbag']:
                rects.append((x1, y1, x2, y2))

        objects = self.tracker.update(rects)

        num_objects = 0

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.classes[class_id]

            if class_name not in ['person', 'backpack', 'handbag']:
                continue

            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            object_id = None
            for id, tracked_centroid in objects.items():
                if np.allclose(centroid, tracked_centroid, atol=1.0):
                    object_id = id
                    break

            if object_id is None:
                continue

            # رسم الصندوق والنص على الإطار
            label = f'{class_name} {confidence:.2f} ID: {object_id}'
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # تسجيل الكائن في قاعدة البيانات إذا كان جديدًا
            if object_id not in self.logged_objects:
                log_detection(object_id, class_name, confidence, x1, y1, x2, y2)
                self.logged_objects.add(object_id)

            num_objects += 1

        # عرض عدد الكائنات المكتشفة على الإطار
        cv2.putText(frame, f'Counting: {num_objects}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        return frame

    def get_detected_boxes(self):
        return self.detected_boxes
