import os
import warnings
import torch
import cv2
import threading
import pyttsx3
import speech_recognition as sr
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys": return {}
    return _orig_getattr(self, name)
torch.nn.Module.__getattr__ = _patched_getattr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"Starting System on: {DEVICE} ({DTYPE})")

print("\nLoading SmolVLM-Instruct (Generative Multimodal AI)...")
model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=DTYPE,
    trust_remote_code=True
).to(DEVICE)

print("Model Ready.\n")

speech_lock = threading.Lock()
is_processing = False
last_captured_image = None

def speak(text):
    with speech_lock:
        print(f"AI: {text}")
        engine = pyttsx3.init()
        engine.setProperty("rate", 160) 
        engine.say(text)
        engine.runAndWait()
        engine.stop()

def listen_to_user():
    recognizer = sr.Recognizer()
    
    recognizer.energy_threshold = 150      
    recognizer.dynamic_energy_threshold = False 
    recognizer.pause_threshold = 2.0       
    
    try:
        with sr.Microphone() as source:
            print("\n🎤 [LISTENING] Calibrating for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            speak("Please ask your question about the scene.")
            
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            
            print("Transcribing voice...")
            return recognizer.recognize_google(audio)
    except Exception as e:
        print(f"Mic Error: {e}")
        return None

def get_ai_response(image, user_prompt):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)


    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2
    )
    
    
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    response = generated_texts[0].split("Assistant:")[-1].strip()
    return response


cap = cv2.VideoCapture(0) 


print("=========================================")
print("  GEN-AI ASSISTIVE VISION SYSTEM")
print("  SPACE -> Capture Scene & Describe")
print("  V     -> Voice Query (Ask a Question)")
print("  ESC   -> Exit Program")
print("=========================================\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.imshow("Assistive Vision Feed", frame)
    key = cv2.waitKey(1)


    if key == 32 and not is_processing:
        is_processing = True
        
        last_captured_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        def describe_task():
            global is_processing
            print("Processing scene description...")
            prompt = "Briefly describe this scene for a blind person. Focus on immediate obstacles."
            ans = get_ai_response(last_captured_image, prompt)
            speak(ans)
            is_processing = False
        
        threading.Thread(target=describe_task).start()


    elif key == ord('v') and not is_processing:
        is_processing = True
        def vqa_task():
            global is_processing
            
            img = last_captured_image if last_captured_image else Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            question = listen_to_user()
            if question:
                speak(f"User asked: {question}")
                ans = get_ai_response(img, question)
                speak(ans)
            else:
                speak("I'm sorry, I didn't hear a question.")
            is_processing = False
            
        threading.Thread(target=vqa_task).start()

    elif key == 27: break

cap.release()
cv2.destroyAllWindows()
