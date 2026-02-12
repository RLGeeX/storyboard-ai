from config import MODEL_NAME
from .utils import client, _save_to_run_folder

def prompt_tool_fn(scene_description: str) -> str:
    """
    Generates a specialized image prompt for a professional storyboard.
    The style is a clean sketch with selective vibrant colors on important objects.
    
    Args:
        scene_description: Visual description of the scene.
    Returns:
        The image generation prompt string.
    """
    prompt = f"""
    You are an expert storyboard artist for high-end cinematic productions.
    
    Input Description: "{scene_description}"
    
    Your Task: Convert this description into a strict image generation prompt for a storyboard frame.
    
    Style Guidelines:
    - Overall Style: Clean professional storyboard sketch, charcoal or pencil linework, high contrast.
    - Background: Mostly white or light gray minimalist sketch.
    - Color Logic: The image must be primarily black and white, BUT you must identify the 1-2 most important objects or focal points in the scene and give them vibrant, saturated colors. This selective coloring should highlight the significance of these elements.
    
    Construct the final prompt using these keywords:
    "professional storyboard style, clean pencil sketch, charcoal outlines, white background, high contrast, {scene_description}, selective color on key elements, minimalist, cinematic composition, no text"
    
    Output: ONLY the final prompt string that will be sent to an AI image generator. Ensure the prompt explicitly describes which parts are colored and in what color.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        result = response.text.strip()
        # Clean up any potential markdown formatting if the model wraps it in quotes or blocks
        result = result.replace('"', '').replace('`', '').strip()
        
        _save_to_run_folder(f"Scene: {scene_description}\nPrompt: {result}\n---\n", "prompts_log.txt", mode="a")
        return result
    except Exception as e:
        return f"Error prompt: {e}"
