import cv2
from PIL import Image

class CameraManager:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            
    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame
        
    def get_pil_image(self, frame=None, max_size=(640, 480)):
        if frame is None:
            ret, frame = self.get_frame()
            if not ret: return None
            
        if max_size:
            frame = cv2.resize(frame, max_size)
            
        # Convert BGR to RGB
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    def release(self):
        self.cap.release()
