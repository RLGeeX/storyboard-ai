from config import MODEL_NAME
from .utils import client, _save_to_run_folder
from google.genai import types
import os

def refine_narration_tool_fn(original_narration: str, image_path: str, video_duration: float = None, global_plan: dict = None) -> str:
    """
    Refines a narration script based on visual contents and the Director's plan.
    
    Args:
        original_narration: The baseline narration text.
        image_path: Path to the image.
        video_duration: Duration in seconds.
        global_plan: The global plan dictionary (from the Director).
    Returns:
        A refined narration string.
    """
    if not client:
        return original_narration

    if not os.path.exists(image_path):
        print(f"Warning: Image not found for narration refinement at {image_path}. Using original.")
        return original_narration

    duration_context = ""
    if video_duration:
        duration_context = f"The visual animation for this scene is {video_duration:.1f} seconds long. Your narration should be descriptively rich and slightly longer than this duration to allow for complete, cinematic storytelling (aim for a natural speaking pace)."

    persona = global_plan.get("narrative_persona", "Professional Storyteller") if global_plan else "Professional Storyteller"
    tone = global_plan.get("tone", "dramatic") if global_plan else "dramatic"

    prompt = f"""
    You are a {persona}. 
    Your voice brings still images to life with the perfect pace and {tone} tone.
    
    Original Narration: "{original_narration}"
    Visual Context: Look at the image. {duration_context}
    
    Your Task: Rewrite the narration to be specific to the visual details while following the {tone} tone.
    
    Requirements:
    1. Persona Alignment: Strictly follow the persona of a {persona}.
    2. Visual Specificity: Mention colors, textures, or key objects seen in the image.
    3. Pacing: Length should allow for a natural speaking pace for {video_duration or 5} seconds.
    4. Cues: Add [dramatic pause], [softly], etc. only where appropriate for the {tone} style. 
    """

    try:
        mime_type = "image/png"
        if image_path.lower().endswith((".jpg", ".jpeg")):
            mime_type = "image/jpeg"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ]
        )
        
        result = response.text.strip()
        _save_to_run_folder(f"Original: {original_narration}\nRefined: {result}\n---\n", "narration_refinement_log.txt", mode="a")
        return result
        
    except Exception as e:
        print(f"Error refining narration: {e}")
        return original_narration
