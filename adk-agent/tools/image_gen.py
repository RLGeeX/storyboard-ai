import os
import time
from google.genai import types
from config import IMAGE_GEN_MODEL
from . import utils

def image_gen_tool_fn(prompt: str, reference_image_path: str = None) -> str:
    """
    Generates a whiteboard animation image using Gemini 2.5 Flash Image.
    
    Args:
        prompt: The specific text prompt to generate an image for.
        reference_image_path: Optional path to a previously generated image for aesthetic consistency.
    Returns:
        The path to the generated image or an error message.
    """
    base_aesthetic = "Professional whiteboard animation style, black line drawing, thick black marker, white background, clean lines, high contrast. The drawing should be clear and can include relevant text or labels as requested."
    full_prompt = f"{base_aesthetic} Subject: {prompt}"
    
    if not utils.client:
        return "Error: GEMINI_API_KEY not configured."

    try:
        contents = [full_prompt]
        
        if reference_image_path and os.path.exists(reference_image_path):
            try:
                with open(reference_image_path, "rb") as f:
                    image_bytes = f.read()
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
            except Exception as e:
                print(f"Warning: Could not read reference image {reference_image_path}: {e}")

        # Use the configured image generation model
        response = utils.client.models.generate_content(
            model=IMAGE_GEN_MODEL,
            contents=contents,
        )
        
        output_path = None
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                timestamp = int(time.time())
                filename = f"generated_image_{timestamp}.png"
                output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, filename) if utils.GLOBAL_OUTPUT_DIR else filename
                image.save(output_path)
                print(f"Image generated and saved to: {output_path}")
                break
            elif part.text:
                print(f"Model response text: {part.text}")
        
        if output_path:
            return output_path
        
        return "Error: No image was generated in the response. Check logs for model text."
    except Exception as e:
        return f"An error occurred during image generation: {str(e)}"
