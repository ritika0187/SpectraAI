"""Small utilities for inspecting a video's audio stream."""

from pathlib import Path
import subprocess

import imageio_ffmpeg


def has_audio_stream(video_path: str | Path) -> bool:
    """Return whether the video container declares at least one audio stream."""

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-i",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    stream_details = f"{result.stdout}\n{result.stderr}".lower()
    return "audio:" in stream_details
