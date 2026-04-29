import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models
# User referred to "flash3", likely meaning the latest Gemini 2.0 Flash
MODEL_NAME = "gemini-2.5-pro" 
# Deep Research Model
DEEP_RESEARCH_MODEL = "deep-research-pro-preview-12-2025"
# Image Generation Model
# gemini-3-pro-image-preview is gated/allowlist-only; switched to GA gemini-2.5-flash-image.
IMAGE_GEN_MODEL = "gemini-2.5-flash-image"
# IMAGE_GEN_MODEL = "gemini-3-pro-image-preview"  # requires preview access
# TTS Model
# gemini-2.5-flash-preview-tts graduated to GA as gemini-2.5-flash-tts (per Vertex AI docs).
TTS_MODEL = "gemini-2.5-flash-tts"
# TTS_MODEL = "gemini-2.5-flash-preview-tts"  # deprecated preview alias
# SAM Segmentation Model URL — overridable via SAM_API_URL env var (set in .env).
# Default is the upstream public Cloud Run endpoint; override to a local server like
# http://localhost:8000/predict (florence2-server) or http://localhost:8001/predict (sam3-server).
SAM_API_URL = os.getenv(
    "SAM_API_URL",
    "https://sam3-app-1040077537378.us-east4.run.app/predict",
)

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")
