from google import genai
from google.genai import types
from PIL import Image
from config import GEMINI_API_KEY, MODEL_NAME, DEEP_RESEARCH_MODEL, IMAGE_GEN_MODEL, TTS_MODEL
import time
import json
import re
import os
import wave
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
    
    if not client:
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
        response = client.models.generate_content(
            model=IMAGE_GEN_MODEL,
            contents=contents,
        )
        
        output_path = None
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                timestamp = int(time.time())
                output_path = f"generated_image_{timestamp}.png"
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

def save_pcm_to_wav(filename: str, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2):
    """
    Saves raw PCM data to a .wav file.
    """
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return True
    except Exception as e:
        print(f"Error saving wave file: {e}")
        return False

def generate_tts_audio_tool_fn(text: str, speaker_one: str = None, speaker_two: str = None) -> str:
    """
    Generates high-quality TTS audio from text using Gemini 2.5 Flash TTS.
    Supports up to 2 speakers for conversational text.
    
    Args:
        text: The text to convert to speech. Use 'Speaker: text' format for multi-speaker.
        speaker_one: Optional name of the first speaker (e.g., 'Joe').
        speaker_two: Optional name of the second speaker (e.g., 'Jane').
    Returns:
        The path to the generated .wav file or an error message.
    """
    if not client:
        return "Error: GEMINI_API_KEY not configured."

    print(f"Generating TTS for: {text[:50]}...")

    try:
        # Determine if we should use MultiSpeakerVoiceConfig
        speech_config = None
        if speaker_one or speaker_two:
            speaker_voice_configs = []
            if speaker_one:
                speaker_voice_configs.append(
                    types.SpeakerVoiceConfig(
                        speaker=speaker_one,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore')
                        )
                    )
                )
            if speaker_two:
                speaker_voice_configs.append(
                    types.SpeakerVoiceConfig(
                        speaker=speaker_two,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck')
                        )
                    )
                )
            
            speech_config = types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_voice_configs
                )
            )
        else:
            # Single speaker default
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore')
                )
            )

        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
            )
        )

        # Extract audio data
        if not response.candidates or not response.candidates[0].content.parts:
            return "Error: No audio generated in response candidates."
        
        audio_part = response.candidates[0].content.parts[0]
        if audio_part.inline_data is None:
            return "Error: No inline audio data found in the response."

        data = audio_part.inline_data.data
        
        timestamp = int(time.time())
        output_path = f"generated_audio_{timestamp}.wav"
        
        if save_pcm_to_wav(output_path, data):
            print(f"TTS audio generated and saved to: {output_path}")
            return output_path
        
        return "Error: Failed to save wave file."

    except Exception as e:
        return f"An error occurred during TTS generation: {str(e)}"
