from config import MODEL_NAME
from .utils import client, _save_to_run_folder

def prompt_tool_fn(scene_description: str, visual_setup: str = "", global_plan: dict = None) -> str:
    """
    Generates an image prompt for a whiteboard animation frame.
    
    The style is: hand-drawn on a clean white background with marker/pen lines,
    like a professional whiteboard animation. Key elements get selective color
    while the rest stays black-and-white line art.
    
    Args:
        scene_description: Visual description of the scene.
        visual_setup: Specific instructions for this frame (from the Director).
        global_plan: The global plan dictionary (from the Director).
    Returns:
        The image generation prompt string.
    """
    tone = global_plan.get("tone", "dramatic") if global_plan else "dramatic"
    visual_style = global_plan.get("visual_style", "Clean Whiteboard Animation") if global_plan else "Clean Whiteboard Animation"
    
    tone_guidance = ""
    if tone == "informative":
        tone_guidance = "The visual should be clear, accurate, and educational. Use clean diagrams, simple icons, and readable layouts. Avoid dramatic exaggeration."
    elif tone == "dramatic":
        tone_guidance = "The visual should be expressive and cinematic. Use dynamic compositions, bold lines, and dramatic framing."
    else:
        tone_guidance = "The visual should be engaging and clear, suitable for the story being told."

    prompt = f"""
    You are an expert whiteboard animation artist.
    
    Your job: Create an image prompt for a WHITEBOARD ANIMATION frame.
    
    WHAT WHITEBOARD ANIMATION LOOKS LIKE:
    - Clean WHITE background (like a dry-erase whiteboard)
    - Simple, quick LINE DRAWINGS using black marker/pen strokes
    - Hand-drawn aesthetic — not photorealistic, not heavily detailed
    - NO shading, NO gradients — flat simple strokes only
    - Like someone is drawing with a marker on a whiteboard in real-time
    - Think: explainer video, educational whiteboard, hand-drawn illustrations
    
    COLOR ENHANCEMENT RULE:
    - The drawing is primarily BLACK lines on WHITE background
    - 1-2 KEY objects or focal areas should have VIBRANT selective color
    - The color should draw attention to the most important element in the scene
    - Everything else stays black-and-white line art
    - Example: A scene about a boy could have the boy in color, everything else in black lines
    
    SCENE DETAILS:
    - Description: "{scene_description}"
    - Visual Setup: "{visual_setup}"
    - Tone: {tone}
    - {tone_guidance}
    
    CONSTRUCT the final image generation prompt following this template:
    "Whiteboard animation frame, hand-drawn with black marker on clean white background, 
    simple line drawing style, [scene content from description and visual_setup], 
    [1-2 key elements] highlighted in vibrant selective color, rest in black-and-white line art, 
    no shading, no gradients, educational explainer style, clear composition, no text"
    
    Output: ONLY the final prompt string. No explanations.
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

