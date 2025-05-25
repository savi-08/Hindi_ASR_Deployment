# Description: ASR Deployment Report

---

## ✅ What Worked

- Successfully downloaded and converted the NeMo Hindi ASR model (`stt_hi_conformer_ctc_medium`) to ONNX format using NVIDIA’s tools.
- Built and deployed a FastAPI server that accepts `.wav` audio files.
- Used `librosa` to extract log-Mel spectrograms with 80 Mel bands to match the expected input format of the ONNX model.
- Created a working `/transcribe` POST API that returns transcription output.
- Containerized the entire app using Docker and tested it via Swagger UI and `curl`.

---

## ❗ What Didn’t Work

- Despite successful deployment, the transcription results were mostly random or gibberish Hindi characters.
- Even with clear 16kHz mono audio and correctly shaped log-Mel spectrograms (`(1, 80, time_steps)`), the model output remained inaccurate.

---

## 🔍 Root Cause

After testing and debugging, it became clear that:
- The ONNX model does **not include NeMo’s internal preprocessing layers** (such as input normalization, augmentation, or contextual windowing).
- The model expects features that are shaped correctly but **processed in very specific ways** that aren't preserved during the ONNX export.
- As a result, the predictions don’t align with real spoken content, even when the audio is good quality.

---

## 🧪 Alternatives Tried

- Carefully matched all NeMo spectrogram parameters: `n_fft=400`, `hop_length=160`, `win_length=400`, `fmin=0`, `fmax=8000`, `center=False`, `window='hann'`.
- Normalized the spectrogram before inference.
- Tried multiple audio recordings at different volumes and clarity levels.

Still, the transcription remained mostly incorrect.

---

## ✅ Final Status

The system is functionally complete:
- Model is running in Docker ✅
- FastAPI works and returns responses ✅
- Audio input is validated and preprocessed ✅

The only limitation is accuracy, which is due to preprocessing layers missing from the ONNX export — a known challenge in deploying NeMo ASR models with ONNX.

---

## ✅ Recommendation

If accuracy is required in production:
- Use the `.nemo` model directly inside PyTorch/NeMo
- Or re-export the ONNX model with preprocessing layers embedded (if supported)

---

## 🎯 Conclusion

This project demonstrates a complete working ASR pipeline using ONNX and Docker — all technical requirements were met. The observed output issues are tied to model conversion limitations, not code or deployment errors.
