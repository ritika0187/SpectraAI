import cv2


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": round(fps, 2),
        "frames": frames,
        "duration": round(duration, 2),
        "resolution": f"{width} × {height}"
    }