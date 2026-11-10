from typing import Any
import cv2


def get_video_frame(video_path: str, frame_number: int = 0) -> Any:
    capture = cv2.VideoCapture(video_path)
    frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    # fix: frame_number - 1 could become -1 for frame 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_total - 1, int(frame_number))))
    has_frame, frame = capture.read()
    capture.release()
    return frame if has_frame else None


def get_video_frame_total(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
