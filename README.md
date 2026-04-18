# 🧠 Gen-AI Assistive Vision System

A real-time **multimodal assistive system** that combines computer vision, speech recognition, and generative AI to help visually impaired users understand and interact with their surroundings.

---

## 🚀 Overview

This project uses a **Vision-Language Model (VLM)** to analyze live camera input and provide:

* 📷 Scene descriptions for navigation
* 🎤 Voice-based question answering about surroundings
* 🔊 Real-time spoken responses

The system is designed to act as an **AI-powered assistive companion**.

---

## ✨ Features

* 🔍 **Real-time Scene Understanding**

  * Captures live camera feed
  * Generates concise descriptions focused on obstacles

* 🗣️ **Voice Interaction**

  * Ask questions about the environment
  * Example: *“Is there a chair in front of me?”*

* 🔊 **Text-to-Speech Output**

  * AI responses are spoken aloud

* ⚡ **Multithreaded Processing**

  * Smooth real-time experience without freezing the camera feed

* 🧠 **Multimodal AI**

  * Combines image + text understanding

---

## 🏗️ Tech Stack

* **Python**
* **PyTorch** – model inference
* **Hugging Face Transformers** – multimodal model loading
* **OpenCV** – video capture
* **SpeechRecognition** – voice input
* **pyttsx3** – offline text-to-speech
* **PIL (Pillow)** – image processing

---

## 🤖 Model Used

* `HuggingFaceTB/SmolVLM-500M-Instruct`

  * Lightweight Vision-Language Model
  * Supports image + text reasoning
  * Optimized for local inference

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/gen-ai-assistive-vision.git
cd gen-ai-assistive-vision
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers opencv-python pillow
pip install speechrecognition pyttsx3
```

### 3. (Optional) Install microphone support

```bash
pip install pyaudio
```

---

## ▶️ Usage

Run the script:

```bash
python main.py
```

### 🎮 Controls

| Key     | Action                          |
| ------- | ------------------------------- |
| `SPACE` | Capture scene & get description |
| `V`     | Ask a voice question            |
| `ESC`   | Exit application                |

---

## 🔄 How It Works

1. 📷 Camera captures a frame using OpenCV
2. 🧠 Image + prompt sent to Vision-Language Model
3. ✍️ Model generates a response
4. 🔊 Response is converted to speech
5. 🎤 User can ask follow-up questions via voice

---

## 🧪 Example Use Cases

* 🚶‍♂️ Indoor navigation assistance
* 🪑 Detecting obstacles (chairs, tables, walls)
* 📦 Understanding surroundings
* ❓ Asking contextual questions about the scene

---

## ⚠️ Limitations

* 🎤 Speech recognition requires internet (Google API)
* 🐢 Slower performance on CPU-only systems
* 🧠 Limited context (single image memory)
* 📉 Short responses due to token limits

---

## 🚀 Future Improvements

* 🔍 Object detection (YOLO integration)
* 📏 Depth estimation for distance awareness
* 🧭 Navigation guidance (left/right instructions)
* 🧠 Larger multimodal models (LLaVA, etc.)
* 🎤 Offline speech recognition (Whisper/Vosk)
* 🖥️ GUI interface

---

## 📌 Project Structure

```
.
├── main.py
├── README.md
```

---


## 📜 License

This project is open-source and available under the MIT License.

---

## 🙌 Acknowledgements

* Hugging Face for pretrained models
* Open-source Python ecosystem
