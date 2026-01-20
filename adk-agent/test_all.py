from tools import research_tool_fn, divider_tool_fn, prompt_tool_fn, image_gen_tool_fn, generate_tts_audio_tool_fn
import time
import json

def test_research():
    context = "The origin of the humble pencil"
    print(f"\n--- Testing Research Tool with context: '{context}' ---")
    print("This may take several minutes. Please wait...")
    
    start_time = time.time()
    result = research_tool_fn(context)
    end_time = time.time()
    
    print(f"\n--- Research Completed in {round(end_time - start_time, 2)} seconds ---")
    print("\nResult Summary (first 500 chars):")
    print("-" * 20)
    print(result[:500] + "...")
    print("-" * 20)
    return result

def test_image_gen():
    print("\n--- Testing Image Generation Tool ---")
    prompt = "A professional whiteboard animation of a coder working on a complex AI project."
    print(f"Generating image for: {prompt}")
    
    image_path = image_gen_tool_fn(prompt)
    
    if image_path and "Error" not in image_path:
        print(f"SUCCESS: Image generated and saved to {image_path}")
        
        # Test consistency
        prompt2 = "The coder successfully completes the AI project and celebrates."
        print(f"Generating second image for consistency: {prompt2}")
        image_path2 = image_gen_tool_fn(prompt2, reference_image_path=image_path)
        
        if image_path2 and "Error" not in image_path2:
            print(f"SUCCESS: Second image saved to {image_path2}")
        else:
            print(f"FAILED: {image_path2}")
    else:
        print(f"FAILED: {image_path}")

def test_divider_and_prompt(research_output=None, test_image=False):
    if research_output is None:
        research_output = """
        Chess is a game played between two opponents on opposite sides of a 64-square board containing 32 pieces. 
        Each player starts with 16 pieces: eight pawns, two knights, two bishops, two rooks, one queen, and one king. 
        The goal of the game is to checkmate the other king. 
        Chess is believed to have originated in India during the Gupta empire, proving to be an early form of chess known as chaturaṅga.
        """
    
    print("\n--- Testing Divider Tool ---")
    scenes = divider_tool_fn(research_output)
    print(f"Generated {len(scenes)} scenes.")
    
    if scenes and isinstance(scenes, list) and len(scenes) > 0:
        print("\n--- Testing Prompt Tool with the first scene ---")
        first_scene = scenes[0]
        description = first_scene.get('description', '')
        if description:
            prompt = prompt_tool_fn(description)
            print(f"Scene Description: {description}")
            print(f"Generated Prompt: {prompt}")
            
            if test_image:
                print("\n--- Testing Image Gen with the generated prompt ---")
                image_gen_tool_fn(prompt)
        else:
            print("First scene logic error: no description found.")
    else:
        print("Divider tool failed to generate scenes or output is not a list.")

def test_tts():
    print("\n--- Testing TTS Generation Tool ---")
    
    # 1. Single Speaker Test
    text_single = "Say cheerfully: Have a wonderful day!"
    print(f"Testing Single Speaker TTS: '{text_single}'")
    audio_path = generate_tts_audio_tool_fn(text_single)
    if audio_path and "Error" not in audio_path:
        print(f"SUCCESS: Single speaker audio saved to {audio_path}")
    else:
        print(f"FAILED: {audio_path}")

    # 2. Multi-Speaker Test
    prompt_multi = """TTS the following conversation between Joe and Jane:
             Joe: How's it going today Jane?
             Jane: Not too bad, how about you?"""
    print("\nTesting Multi-Speaker TTS (Joe and Jane)")
    audio_path_multi = generate_tts_audio_tool_fn(prompt_multi, speaker_one="Joe", speaker_two="Jane")
    if audio_path_multi and "Error" not in audio_path_multi:
        print(f"SUCCESS: Multi-speaker audio saved to {audio_path_multi}")
    else:
        print(f"FAILED: {audio_path_multi}")

if __name__ == "__main__":
    import sys
    
    print("Select test to run:")
    print("1. Research + Divider + Prompt + Image (Full Flow)")
    print("2. Divider + Prompt (Fast, using mock data)")
    print("3. Standalone Image Generation Test")
    print("4. Standalone TTS Generation Test")
    
    choice = input("Enter choice (1/2/3/4): ")
    
    if choice == '1':
        res = test_research()
        test_divider_and_prompt(res, test_image=True)
    elif choice == '2':
        test_divider_and_prompt()
    elif choice == '3':
        test_image_gen()
    elif choice == '4':
        test_tts()
    else:
        print("Invalid choice.")
