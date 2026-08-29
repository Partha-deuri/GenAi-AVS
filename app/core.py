import cv2
import threading
from .camera import CameraManager
from .audio import AudioManager
from .vision import VisionModel

class AssistiveVisionApp:
    def __init__(self):
        self.camera = CameraManager()
        self.audio = AudioManager()
        
        self.audio.speak_async("Connecting to Gemini API. Please wait.")
        self.vision = VisionModel() 
        
        self.is_processing = False
        self.running = True
        self.last_frame = None
        self.last_pil_image = None
        
        self.audio.speak_async("System ready. You can say 'assistant' to ask a question, or press space to describe the scene.")

    def run_describe_task(self):
        if self.is_processing or self.last_pil_image is None: return
        self.is_processing = True
        self.audio.speak_async("Capturing scene...")
        
        img = self.last_pil_image.copy()
        
        def task():
            print("Processing scene description...")
            prompt = "Briefly describe this scene for a blind person. Focus on immediate obstacles."
            ans = self.vision.get_ai_response(img, prompt)
            self.audio.speak(ans)
            self.is_processing = False
            
        threading.Thread(target=task, daemon=True).start()

    def run_vqa_task(self):
        if self.is_processing or self.last_pil_image is None: return
        self.is_processing = True
        
        img = self.last_pil_image.copy()
        
        def task():
            question = self.audio.listen_to_user(prompt="Please ask your question about the scene.")
            if question:
                self.audio.speak(f"You asked: {question}")
                ans = self.vision.get_ai_response(img, question)
                self.audio.speak(ans)
            else:
                self.audio.speak("I'm sorry, I didn't catch that.")
            self.is_processing = False
            
        threading.Thread(target=task, daemon=True).start()
        
    def wake_word_loop(self):
        # Added similar sounding words for 'vision' to help Google STT
        wake_words = ["assistant", "hey assistant", "vision", "vison", "visions", "bision"]
        while self.running:
            if not self.is_processing:
                detected = self.audio.listen_for_wakeword(wake_words)
                if detected and self.running and not self.is_processing:
                    self.audio.speak_async("Yes?")
                    self.run_vqa_task()

    def start(self):
        threading.Thread(target=self.wake_word_loop, daemon=True).start()
        
        print("=========================================")
        print("  GEN-AI ASSISTIVE VISION SYSTEM")
        print("  SPACE -> Capture Scene & Describe")
        print("  V     -> Voice Query (Ask a Question)")
        print("  ESC   -> Exit Program")
        print("  WAKE WORD -> Say 'assistant' or 'vision'")
        print("=========================================\n")

        while self.running:
            ret, frame = self.camera.get_frame()
            if not ret: break
            
            self.last_frame = frame
            self.last_pil_image = self.camera.get_pil_image(frame)
            
            cv2.imshow("Assistive Vision Feed", frame)
            key = cv2.waitKey(1)

            if key == 32: # Space
                self.run_describe_task()
            elif key == ord('v'):
                self.run_vqa_task()
            elif key == 27: # ESC
                self.running = False
                break

        self.camera.release()
        cv2.destroyAllWindows()
