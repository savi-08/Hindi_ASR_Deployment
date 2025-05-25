from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import soundfile as sf
import librosa
import os
import onnxruntime as ort

app = FastAPI()

# Load ONNX model
onnx_model_path = "asr_model.onnx"
if not os.path.exists(onnx_model_path):
    raise RuntimeError("ONNX model not found!")

session = ort.InferenceSession(onnx_model_path)

# Hindi character labels (example from stt_hi_conformer_ctc model)
labels = [
    " ", "अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ", "अं", "क", "ख", "ग", "घ", "च", "छ", "ज", "झ",
    "ट", "ठ", "ड", "ढ", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष",
    "स", "ह", "ळ", "क्ष", "ज्ञ", "ँ", "ं", "ः", "़", "ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "्", "ॉ", "ॊ", "ॆ"
] + ["<blank>"]

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    # Save uploaded file temporarily
    audio_bytes = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    # Load and resample audio to 16kHz mono
    signal, sr = librosa.load("temp.wav", sr=16000, mono=True)
    os.remove("temp.wav")

    # Normalize audio
    signal = signal / np.abs(signal).max()

    # Extract log-Mel spectrogram with NeMo-matching parameters
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=16000,
        n_fft=400,
        hop_length=160,
        win_length=400,
        window="hann",
        center=False,
        power=2.0,
        n_mels=80,
        fmin=0,
        fmax=8000
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0)

    # Normalize the spectrogram
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-9)

    # Reshape to (1, 80, time_steps)
    log_mel_spec = log_mel_spec[np.newaxis, :, :].astype(np.float32)

    # Length for ONNX input
    length = np.array([log_mel_spec.shape[2]], dtype=np.int64)

    # Debug info
    print("🧪 log_mel_spec shape:", log_mel_spec.shape)
    print("🧪 log_mel_spec dtype:", log_mel_spec.dtype)
    print("🧪 length:", length)

    # Prepare ONNX inputs
    inputs = {
        session.get_inputs()[0].name: log_mel_spec,
        session.get_inputs()[1].name: length
    }

    # Run inference
    outputs = session.run(None, inputs)[0]
    pred = np.argmax(outputs, axis=-1)[0]

    # Decode prediction (greedy CTC decoding)
    decoded = ""
    last = None
    for i in pred:
        if i != last and i < len(labels) - 1:
            decoded += labels[i]
        last = i

    return {"transcription": decoded.strip()}
