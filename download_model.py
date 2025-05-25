from nemo.collections.asr.models import EncDecCTCModel

print("Loading local .nemo model...")
model = EncDecCTCModel.restore_from("stt_hi_conformer_ctc_medium.nemo")

# Export to ONNX
print("Exporting model to ONNX format...")
model.export("asr_model.onnx")

print("Model exported successfully as 'asr_model.onnx'")
