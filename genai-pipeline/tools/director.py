import json
import re
from typing import List, Dict, Any
from config import MODEL_NAME
from .utils import client, _save_to_run_folder

def director_tool_fn(user_instructions: str, research_material: str = None) -> Dict[str, Any]:
    """
    Acts as the Video Director & Writer — plans the entire video journey.
    
    The Director decides how many scenes are needed, writes the narration script
    for each scene (as a storyteller, not just descriptions), plans the visual
    setup, and defines the overall narrative arc.
    
    Args:
        user_instructions: The user's original topic/instructions.
        research_material: Optional detailed research report to incorporate.
    Returns:
        A dictionary with 'global_plan' and 'scenes'.
    """
    
    # If research material is provided, include it; otherwise just use instructions
    research_block = ""
    if research_material and research_material != user_instructions:
        research_block = f"""
    
    Deep Research Material (USE THIS — it contains rich, detailed information that MUST be woven into your narration):
    ---
    {research_material}
    ---
    """

    prompt = f"""
    You are an award-winning Video Director, Writer, and Storyteller.
    You are planning a whiteboard animation video. Your job is to craft the ENTIRE video — 
    the narrative arc, the script, and the visual direction for every single scene.
    
    User's Topic / Instructions:
    "{user_instructions}"
    {research_block}
    
    YOUR TASK — Plan the complete video:
    
    STEP 1: Analyze the topic and decide:
    - What TONE fits? (informative, dramatic, playful, sad, etc.)
    - What is the NARRATIVE ARC? (beginning hook → build-up → climax → resolution)
    - Who is narrating? (a professional explainer, a storyteller, a historian, etc.)
    - How many scenes are needed? (CRITICAL: Adjust this based on user instructions. For a "quick" or "fast" video, strict limit of 2-3 scenes. For "detailed", you may use more.)
    
    STEP 2: For EACH scene, you must provide:
    - 'scene_number': Sequential number
    - 'summary': A 1-line summary of what this scene accomplishes in the narrative arc (e.g., "Introduces the boy's daily routine and sets up the deception")
    - 'narration': The FULL spoken script for this scene. THIS IS THE MOST IMPORTANT PART.
      * Write it as a STORY, not a description. Use vivid language, dialogue, tension, emotion.
      * If research material was provided, weave the key facts/details INTO the narrative naturally.
      * Each scene's narration should be 2-5 sentences, speakable in 10-20 seconds.
      * The narration should flow naturally from scene to scene — it's one continuous story.
    - 'description': Visual description for the image generator (what should be DRAWN in this frame)
    - 'visual_setup': Specific visual direction for this frame (composition, key elements, focal points)
    - 'search_query': (OPTIONAL) If this scene features a specific real-world person, historical figure, or landmark (e.g., "5th president of France"), provide a search query here so the system can find a reference photo. Leave empty if a generic drawing is fine.
    - 'text_overlay': (OPTIONAL) If you want specific impactful text visually rendered IN the scene (e.g., a date, a powerful quote, a key stat), specify it here along with desired styling (e.g. "1969 in bold sans-serif").
    - 'key_information': Any critical facts/data from the research that this scene must convey
    - 'emotional_beat': The emotional tone of this specific scene (e.g., "playful", "tense", "triumphant", "cautionary")
    
    CRITICAL RULES:
    - ATTRACTIVE PACING & TONE: You MUST detect pacing instructions from the user (e.g. "quick", "fast", "detailed", "sad"). Force the pacing and styling to obey these modifiers exactly.
    - You are the WRITER. The narration you write IS the final script. Make it compelling.
    - If research material is provided, DO NOT lose the information. Incorporate key facts into the story.
    - Each scene narration must be a natural continuation of the previous — it's ONE story, not isolated descriptions.
    - The narration should sound like someone TELLING a story, not reading a textbook.
    - Think about pacing: start engaging, build tension/interest, deliver the payoff, end memorably.
    
    Output Format (Strict JSON):
    {{
      "global_plan": {{
        "title": "Video title",
        "tone": "informative" | "dramatic" | "educational" | "cautionary",
        "narrative_persona": "e.g., Wise Storyteller, Professional Documentary Narrator, Enthusiastic Teacher",
        "visual_style": "e.g., Clean Whiteboard Animation, Cinematic Historical Sketch",
        "pacing": "e.g., steady/educational, building/dramatic, fast/action",
        "narrative_arc": "Brief description of the story's arc from opening to conclusion",
        "target_audience": "e.g., general public, students, investors",
        "total_scenes": <number>
      }},
      "scenes": [
        {{
          "scene_number": 1,
          "summary": "...",
          "narration": "...",
          "description": "...",
          "visual_setup": "...",
          "search_query": "...",
          "text_overlay": "...",
          "key_information": "...",
          "emotional_beat": "..."
        }},
        ...
      ]
    }}
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
        _save_to_run_folder(json.dumps(result, indent=2), "video_plan.json")
        return result
    except Exception as e:
        print(f"Error in director_tool: {e}")
        # Fallback to a basic structure if parsing fails
        return {
            "global_plan": {
                "title": "Untitled Video",
                "tone": "informative", 
                "narrative_persona": "Professional Storyteller", 
                "visual_style": "Clean Whiteboard Animation", 
                "pacing": "steady",
                "narrative_arc": "Linear exploration of the topic",
                "target_audience": "general public",
                "total_scenes": 1
            },
            "scenes": [{
                "scene_number": 1,
                "summary": "Error parsing",
                "description": "Error parsing", 
                "narration": "Error parsing", 
                "visual_setup": "Simple sketch",
                "search_query": "",
                "text_overlay": "",
                "key_information": "",
                "emotional_beat": "neutral"
            }]
        }
