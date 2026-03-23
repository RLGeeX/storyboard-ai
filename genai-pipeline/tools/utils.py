import os
import wave
from google import genai
from config import GEMINI_API_KEY

# Setup GenAI Client
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# Global Output Management
GLOBAL_OUTPUT_DIR = None

def set_output_dir(path: str):
    """Sets the directory where all tool outputs will be saved."""
    global GLOBAL_OUTPUT_DIR
    GLOBAL_OUTPUT_DIR = path
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print(f"Output directory set to: {path}")

def _save_to_run_folder(content: str, filename: str, mode: str = "w"):
    """Helper to save content to the current run folder if enabled."""
    if GLOBAL_OUTPUT_DIR:
        full_path = os.path.join(GLOBAL_OUTPUT_DIR, filename)
        try:
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(content)
            return full_path
        except Exception as e:
            print(f"Error saving to {filename}: {e}")
    return None

def save_pcm_to_wav(filename: str, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2):
    """
    Saves raw PCM data to a .wav file.
    """
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return True
    except Exception as e:
        print(f"Error saving wave file: {e}")
        return False

def get_video_duration(video_path: str) -> float:
    """
    Returns the duration of a video file in seconds.
    """
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        return duration
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0.0
