import torch
import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO

class SinglePersonTracker:
  
    def __init__(self, model_path: str):
       
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Model loading error: {str(e)}")
        
        self.tracked_person = None
        self.tracking_started = False
        self.persons_records = []
        self.trajectory = []
        self.user_click = (None, None)
        self.prev_frame = None
        self.lost_person = False
        self.margin = 10

    def get_user_choice(self, frame):
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.user_click = (x, y)
                cv2.destroyAllWindows()

        cv2.imshow("Pick a person to track", frame)
        cv2.setMouseCallback("Pick a person to track", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def find_closest_person(self, cords):
       
        x1, y1, x2, y2,_ = cords
        center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
        min_distance = float('inf')
        for idx, person in enumerate(self.persons_records):
            [person_x1, person_y1, person_x2, person_y2, _] = person['human']
            person_center = ((person_x1 + person_x2) / 2, (person_y1 + person_y2) / 2)
            euclidean_distance = np.sqrt((center_point[0] - person_center[0]) ** 2 + (center_point[1] - person_center[1]) ** 2)
            if euclidean_distance < min_distance:
                min_distance = euclidean_distance
                id = idx
        return id if min_distance < 100 else None
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
       
        try:
            img = frame[:, :, ::-1]  # BGR to RGB
            self.results = self.model(img)
            
            detections = []
            for boxs in self.results[0].boxes:
                if boxs.cls[0] ==0:  # person class 
                    detections.append((
                        int(boxs.xyxy[0][0]), int(boxs.xyxy[0][1]),  # x1, y1
                        int(boxs.xyxy[0][2]), int(boxs.xyxy[0][3]),  # x2, y2
                        float(boxs.conf[0])                  # confidence
                    ))
            return detections
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return []
    
    def track_single_person(self, frame: np.ndarray) -> np.ndarray:
        
        if self.lost_person: 
           
            x1, y1, x2, y2, _ = self.tracked_person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Lost", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if self.trajectory:
                for i in range(1, len(self.trajectory)):
                    cv2.line(frame, self.trajectory[i-1], self.trajectory[i],
                            (0, 0, 255), thickness=3)
            return frame
        
        while self.user_click[0] is None:
            self.get_user_choice(frame)
        detections = self.detect_persons(frame)
        
        
        if not self.tracking_started and detections:
            min_dist = float('inf')
            best_det = None
            
            for det in detections:
                x1, y1, x2, y2, _ = det
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                dist = np.sqrt((center[0]-self.user_click[0])**2 + (center[1]-self.user_click[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_det = det
            
            if best_det:
                self.tracked_person = best_det
                self.tracking_started = True
        
        elif self.tracking_started and detections:
            min_dist = float('inf')
            best_det = None
            x1_prev, y1_prev, x2_prev, y2_prev, _ = self.tracked_person

            for det in detections:
                x1, y1, x2, y2, conf = det
                diff_sum = sum([abs(x1-x1_prev), abs(y1-y1_prev), abs(x2-x2_prev), abs(y2-y2_prev)])
                if diff_sum < min_dist:  
                    min_dist = diff_sum
                    best_det = det
            if best_det and min_dist < 120: 
                self.tracked_person = best_det
            else:
                self.lost_person = True
        
        if self.tracked_person:
            x1, y1, x2, y2, _ = self.tracked_person
            center = ((x1 + x2) // 2, (y1 + y2) // 2)      
            
            if x1 >= 0 + self.margin and y1 >= 0 + self.margin and x2 + self.margin <= self.results[0].orig_shape[1] and y2 + self.margin <= frame.shape[0]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Tracking", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                self.lost_person = True
            # Draw trajectory
            self.trajectory.append(center)
            if len(self.trajectory) > 1:
                for i in range(1, len(self.trajectory)):
                    cv2.line(frame, self.trajectory[i-1], self.trajectory[i],
                            (0, 0, 255), thickness=3)
        
        return frame

