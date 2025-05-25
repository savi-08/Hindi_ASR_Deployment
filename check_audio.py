import soundfile as sf

filename = "final_audio.wav"  
data, samplerate = sf.read(filename)

print("✅ Sample Rate:", samplerate)
print("✅ Duration:", round(len(data) / samplerate, 2), "seconds")
print("✅ Channels:", 1 if len(data.shape) == 1 else data.shape[1])
