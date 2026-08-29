import threading
import pyttsx3
import speech_recognition as sr
import time

class AudioManager:
    def __init__(self):
        self.speech_lock = threading.Lock()
        
    def speak(self, text):
        with self.speech_lock:
            print(f"AI: {text}")
            engine = pyttsx3.init()
            engine.setProperty("rate", 160) 
            engine.say(text)
            engine.runAndWait()
            engine.stop()

    def speak_async(self, text):
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

    def listen_to_user(self, prompt=None, timeout=10, phrase_time_limit=10):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 150      
        recognizer.dynamic_energy_threshold = False 
        recognizer.pause_threshold = 2.0       
        
        try:
            with sr.Microphone() as source:
                print("\n🎤 [LISTENING] Calibrating for background noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                if prompt:
                    self.speak(prompt)
                
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                print("Transcribing voice...")
                return recognizer.recognize_google(audio).lower()
        except sr.WaitTimeoutError:
            print("Mic Error: Timed out listening.")
            return None
        except sr.UnknownValueError:
            print("Mic Error: Could not understand audio.")
            return None
        except Exception as e:
            print(f"Mic Error: {e}")
            return None

    def listen_for_wakeword(self, wakewords=["assistant", "hey assistant", "vision"]):
        """
        Continuously listens for a wakeword. 
        """
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 150
        recognizer.dynamic_energy_threshold = True
        
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                while True:
                    try:
                        # short phrase time limit for wake word
                        audio = recognizer.listen(source, phrase_time_limit=3)
                        text = recognizer.recognize_google(audio).lower()
                        for w in wakewords:
                            if w in text:
                                return True
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except Exception as e:
                        time.sleep(1)
        except Exception as e:
            print(f"Wake word listener failed: {e}")
            return False
