from config import MODEL_NAME
from .utils import client, _save_to_run_folder

def prompt_tool_fn(scene_description: str, visual_setup: str = "", global_plan: dict = None) -> str:
    """
    Generates a specialized image prompt for a professional storyboard.
    Adapts based on the Director's planning.
    
    Args:
        scene_description: Visual description of the scene.
        visual_setup: Specific instructions for this frame (from the Director).
        global_plan: The global plan dictionary (from the Director).
    Returns:
        The image generation prompt string.
    """
    tone = global_plan.get("tone", "dramatic") if global_plan else "dramatic"
    visual_style = global_plan.get("visual_style", "Cinematic Storyboard") if global_plan else "Cinematic Storyboard"
    
    realism_instr = ""
    if tone == "informative":
        realism_instr = "Ensure the visual is accurate, informative, and realistic for a documentary. Avoid cinematic exaggeration. Use clear symbols or technical details where specified."

    prompt = f"""
    You are an expert storyboard artist and visual director.
    
    Global Style: {visual_style}
    Tone: {tone}
    Scene Logic: {realism_instr}
    
    Input Description: "{scene_description}"
    Scene Setup: "{visual_setup}"
    
    Your Task: Convert this into a professional storyboard prompt.
    
    Style Guidelines:
    - Overall Style: {visual_style} sketch, charcoal/pencil texture, white background.
    - Setup: Follow the 'Scene Setup' exactly (e.g., charts, realistic maps, specific historical framing).
    - Color Logic: Primarily black and white. 1-2 critical focal points in vibrant selective color.
    
    Construct the final prompt:
    "Professional {visual_style} style, charcoal texture, detailed pencil sketch, white background, {scene_description}, {visual_setup}, focal points in vibrant selective color, grayscale surroundings, minimalist background, cinematic and accurate composition, no text"
    
    Output: ONLY the final prompt string.
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
