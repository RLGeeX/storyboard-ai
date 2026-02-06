from config import MODEL_NAME
from .utils import client, _save_to_run_folder

def prompt_tool_fn(scene_description: str) -> str:
    """
    Generates a specialized image prompt for whiteboard animation.
    
    Args:
        scene_description: Visual description of the scene.
    Returns:
        The image generation prompt string.
    """
    prompt = f"""
    You are an expert prompt engineer for a whiteboard animation AI.
    
    Input Description: "{scene_description}"
    
    Your Task: Convert this description into a strict image generation prompt.
    
    Construct the prompt using these specific keywords:
    "whiteboard animation style, simple black line drawing, thick black marker, white background, minimalist sketch, high contrast, {scene_description}, no text, no shading, no colors, clean lines"
    
    Output: ONLY the final prompt string.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        result = response.text.strip()
        _save_to_run_folder(f"Scene: {scene_description}\nPrompt: {result}\n---\n", "prompts_log.txt", mode="a")
        return result
    except Exception as e:
        return f"Error prompt: {e}"
