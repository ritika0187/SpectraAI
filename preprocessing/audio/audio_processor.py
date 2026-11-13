import os
import subprocess
import tempfile

import imageio_ffmpeg
import librosa
import numpy as np


class AudioProcessor:

    def extract_audio(self, video_path):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:
            wav_path = tmp.name

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg,
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            wav_path,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            if os.path.exists(wav_path):
                os.remove(wav_path)

            raise RuntimeError(
                f"Audio extraction failed:\n{result.stderr}"
            )

        return wav_path

    def load_audio(self, wav_path):
        audio, sr = librosa.load(
            wav_path,
            sr=16000,
            mono=True
        )

        return audio, sr

    def compute_waveform(self, audio, max_points=2500):

        if len(audio) <= max_points:
            return audio

        indices = np.linspace(
            0,
            len(audio) - 1,
            max_points
        ).astype(int)

        return audio[indices]

    def compute_mel_spectrogram(self, audio, sr):

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128,
            fmax=8000
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        return mel_db

    def process(self, video_path):

        wav_path = self.extract_audio(video_path)

        try:
            audio, sr = self.load_audio(wav_path)

            waveform = self.compute_waveform(audio)

            mel_spectrogram = self.compute_mel_spectrogram(
                audio,
                sr
            )

            return {
                "audio": audio,
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "waveform": waveform,
                "mel_spectrogram": mel_spectrogram,
            }

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)