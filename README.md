# Hindi ASR API Deployment using NVIDIA NeMo and ONNX

This project is a FastAPI-based Automatic Speech Recognition (ASR) system for Hindi audio. It uses a pre-trained model from NVIDIA NeMo (`stt_hi_conformer_ctc_medium`) which was converted to ONNX format and deployed using Docker.

---

## 🔧 Features

- Accepts `.wav` audio files (16 kHz, mono, 5–10 seconds)
- Returns transcribed Hindi text
- Built with FastAPI and ONNX Runtime
- Fully containerized with Docker

---

## 🧰 Tech Stack

- Python
- FastAPI
- Librosa
- ONNX Runtime
- Docker

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd <repo-folder>
2. Build Docker Image
bash
Copy
Edit
docker build -t asr-api .
3. Run the API
bash
Copy
Edit
docker run -p 8000:8000 asr-api
The API will be live at:
http://localhost:8000/docs

🎙️ How to Use the API
🔘 Using Swagger UI
Go to http://localhost:8000/docs

Click on /transcribe → "Try it out"

Upload a .wav file (mono, 16kHz)

Click Execute to get the transcription

🔘 Using curl
bash
Copy
Edit
curl -X 'POST' \
  'http://localhost:8000/transcribe' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@fixed_audio.wav'
📁 Project Structure
File	Description
main.py	FastAPI app for handling audio and model inference
Dockerfile	Docker setup to containerize the app
download_model.py	Downloads and converts NeMo model to ONNX
asr_model.onnx	Pre-trained and converted Hindi ASR model
README.md	Instructions to run and use the API
Description.md	Explanation of issues, limitations, and learnings
requirements.txt	Python dependencies for the app

---

## 📷 Screenshots

### Swagger UI

![Swagger UI](screenshot_swagger.png)

### Transcription Output

![Transcription Result](screenshot_result.png)


## 📥 Model Files

Due to GitHub’s 100MB file size limit, the ONNX and `.nemo` models are **not included** in this repo.

You must download them manually using this script:

```bash
python download_model.py

⚙️ Requirements
Audio file must be:

.wav format

Mono (1 channel)

16000 Hz sample rate

Duration between 5 to 10 seconds

📚 Credits
NVIDIA NeMo: https://github.com/NVIDIA/NeMo