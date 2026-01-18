from google import genai
from config import GEMINI_API_KEY, MODEL_NAME, DEEP_RESEARCH_MODEL
import time
import json
import re
from typing import List, Dict, Any

# Setup GenAI Client
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

def research_tool_fn(context: str) -> str:
    """
    Performs end-to-end research on the given context using Gemini Deep Research.
    
    Args:
        context: The topic to research.
    Returns:
        A detailed research report string.
    """
    print(f"Starting Deep Research for: {context}")
    
    if not client:
        return "Error: GEMINI_API_KEY not configured."

    try:
        interaction = client.interactions.create(
            input=f"Research detailed information about: {context}",
            agent=DEEP_RESEARCH_MODEL,
            background=True
        )
        
        print(f"Research started: {interaction.id}")
        
        while True:
            interaction = client.interactions.get(interaction.id)
            if interaction.status == "completed":
                if interaction.outputs:
                    return interaction.outputs[-1].text
                return "Research completed with no output."
            elif interaction.status == "failed":
                return f"Research failed: {interaction.error}"
            
            time.sleep(10)

    except Exception as e:
        return f"An error occurred during research: {str(e)}"

def divider_tool_fn(research_output: str) -> List[Dict[str, Any]]:
    """
    Divides research into storyboard scenes.
    
    Args:
        research_output: The detailed research report text.
    Returns:
        A list of scene dictionaries.
    """
    prompt = f"""
    You are a professional storyboard director. 
    Analyze the following research and break it down into a sequence of scenes for a whiteboard animation video.
    
    Research Material:
    ---
    {research_output}
    ---
    
    Goal: Create a highly engaging, visual narrative.
    
    Requirements:
    1. Break down the content into granular scenes (aim for 1 scene per concept/sentence).
    2. For each scene, write:
       - 'narration': The script to be spoken.
       - 'description': A SIMPLE visual description for a whiteboard drawing.
    
    The 'description' MUST be:
    - Visual and concrete (e.g., "A specific drawing of X doing Y").
    - Suitable for a simple black-and-white line drawing.
    - Avoid abstract concepts like "complexity" without describing HOW to draw it.
    
    Output Format: JSON list of objects.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in divider_tool: {e}")
        try:
             json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
             if json_match:
                 return json.loads(json_match.group(0))
        except:
            pass
        return [{"description": "Error parsing", "narration": "Error parsing"}]

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
        return response.text.strip()
    except Exception as e:
        return f"Error prompt: {e}"
