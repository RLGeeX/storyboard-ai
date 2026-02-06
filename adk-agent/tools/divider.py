import json
import re
from typing import List, Dict, Any
from config import MODEL_NAME
from .utils import client, _save_to_run_folder

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
        result = json.loads(response.text)
        _save_to_run_folder(json.dumps(result, indent=2), "scenes.json")
        return result
    except Exception as e:
        print(f"Error in divider_tool: {e}")
        try:
             json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
             if json_match:
                 return json.loads(json_match.group(0))
        except:
            pass
        return [{"description": "Error parsing", "narration": "Error parsing"}]
