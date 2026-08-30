# Gen-AI Assistive Vision System

A real-time, voice-activated AI assistant designed specifically to help visually impaired individuals understand their surroundings. The system uses a webcam to capture the environment and leverages the lightning-fast **Google Gemini Vision API** to describe scenes and answer user questions audibly.

## Features
- **Continuous Hazard Detection (YOLO):** A local, ultra-fast YOLO nano model constantly scans for immediate physical hazards (people, cars, bicycles, etc.) and warns you instantly.
- **Smart Cooldown System:** Hazard warnings have a customizable cooldown (default 20 seconds) to prevent repetitive audio spam.
- **Real-Time Scene Description:** Instantly get a brief, audible description of the room via Gemini.
- **Voice Question Answering (VQA):** Ask specific questions about what the camera sees (e.g., "Is there a person in front of me?", "What color is the cup on the table?").
- **Hands-Free Wake Word:** Continuously listens in the background. Simply say **"assistant"** or **"vision"** to activate the system without touching the keyboard.
- **Lightning Fast:** Powered by `gemini-3.5-flash-lite`, generating intelligent, conversational responses in seconds.
- **Clean Audio Feedback:** Natural English text-to-speech without reading awkward markdown symbols or AI formatting.

## Prerequisites
- A working **Webcam**
- A working **Microphone**
- Python 3.9+
- A [Google Gemini API Key](https://aistudio.google.com/)

## Installation & Setup

1. **Clone the repository (or download the folder):**
   ```bash
   git clone <your-repo-url>
   cd GenAi-AVS
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the API Key:**
   - Create a file named `.env` in the root directory.
   - Add your Gemini API key to the file like this:
     ```env
     GEMINI_API_KEY=your_actual_api_key_here
     ```

## Usage

Start the system by running the main script:
```bash
python main.py
```

### Controls & Interaction

| Action | How to Trigger | What it Does |
| :--- | :--- | :--- |
| **Wake Word** | Say `"assistant"` or `"vision"` | Activates the Voice Question Answering mode hands-free. Wait for the system to say "Yes?", then ask your question. |
| **Voice Query** | Press `V` | Same as the Wake Word. Turns on the mic to listen to your question about the current scene. |
| **Describe Scene** | Press `SPACE` | Instantly takes a picture and provides a general description of the immediate environment and obstacles. |
| **Toggle Hazards** | Press `T` | Turns the continuous YOLO hazard warnings ON or OFF. Useful for crowded areas. |
| **Adjust Cooldown** | Press `+` or `-` | Increases or decreases the delay between repeated hazard warnings. |
| **Exit** | Press `ESC` | Closes the camera and exits the program. |

## Architecture
- `app/camera.py`: Manages webcam initialization and image formatting.
- `app/audio.py`: Manages local Speech-to-Text and Text-to-Speech interactions, including background wake word monitoring.
- `app/detector.py`: Runs a lightweight YOLO model to silently identify physical hazards in the background.
- `app/vision.py`: Connects to the Google Gemini Vision API to process images and user prompts.
- `app/core.py`: The central controller that manages state and multi-threaded tasks between components.
- `main.py`: The application entry point.

---
*Built with Python, OpenCV, SpeechRecognition, pyttsx3, Ultralytics (YOLO), and Google GenAI.*
