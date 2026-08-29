import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import logging

# Suppress google-genai warnings about Automatic Function Calling
logging.getLogger("google").setLevel(logging.ERROR)

load_dotenv()

class VisionModel:
    def __init__(self, model_id="gemini-3.5-flash-lite"):
        self.model_id = model_id
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            print("[ERROR] GEMINI_API_KEY not set in .env file!")
            self.client = None
            return
            
        print(f"Connecting to Gemini API (Model: {model_id})...")
        self.client = genai.Client(api_key=api_key)
        print("Gemini API Ready.\n")

    def get_ai_response(self, image, user_prompt):
        if not self.client:
            return "Error: API Key is missing. Please check your .env file."
            
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[image, user_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    system_instruction="You are an assistive vision AI for a visually impaired person. Always answer in conversational, natural English. Do NOT use any markdown formatting, asterisks (**), bold text, or bullet points. Keep your answers brief but complete. Output plain text only."
                )
            )
            return response.text.strip()
        except Exception as e:
            import traceback
            print(f"[ERROR] Gemini API call failed: {e}")
            traceback.print_exc()
            return "Sorry, I couldn't reach the online AI."
