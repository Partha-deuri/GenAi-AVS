import time
from ultralytics import YOLO

class HazardDetector:
    def __init__(self, cooldown_seconds=5):
        print("Loading YOLO hazard detection model...")
        # 'yolov8n.pt' is the nano model, which is ultra-fast and perfect for weak devices.
        # It will automatically download on the first run.
        self.model = YOLO('yolov8n.pt')
        self.cooldown_seconds = cooldown_seconds
        self.last_warning_time = {}
        
        # Define COCO classes that are considered priority hazards/obstacles
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 11: stop sign, 15: cat, 16: dog
        self.hazard_classes = {0, 1, 2, 3, 5, 7, 11, 15, 16}

    def detect_hazards(self, frame):
        """
        Runs YOLO on the frame and returns a list of warning strings if hazards are detected
        and their cooldown has expired.
        """
        warnings = []
        
        # Run inference (verbose=False to avoid spamming the console)
        results = self.model(frame, verbose=False)
        
        current_time = time.time()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                
                # Check if the detected object is a priority hazard
                if cls_id in self.hazard_classes:
                    
                    # Optional: We could check bounding box size here to only warn 
                    # if the object is close (large area), but for now, we warn on sight.
                    class_name = self.model.names[cls_id]
                    
                    # Check cooldown for this specific object type
                    last_time = self.last_warning_time.get(class_name, 0)
                    if current_time - last_time > self.cooldown_seconds:
                        warnings.append(f"Warning, {class_name} ahead.")
                        self.last_warning_time[class_name] = current_time
                        
        return warnings
