from preprocessing.audio.audio_processor import AudioProcessor

VIDEO_PATH = "/Users/ritikakhanna/Desktop/sample.mov"
processor = AudioProcessor()

result = processor.process(VIDEO_PATH)

print("Audio samples:", len(result["audio"]))
print("Sample rate:", result["sample_rate"])
print("Duration:", round(result["duration"], 2), "seconds")
print("Waveform points:", len(result["waveform"]))
print("Mel shape:", result["mel_spectrogram"].shape)
print("Waveform min:", result["waveform"].min())
print("Waveform max:", result["waveform"].max())