import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models
# User referred to "flash3", likely meaning the latest Gemini 2.0 Flash
MODEL_NAME = "gemini-2.5-flash" 
# Deep Research Model
DEEP_RESEARCH_MODEL = "deep-research-pro-preview-12-2025"

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")
