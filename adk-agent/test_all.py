from tools import research_tool_fn, divider_tool_fn, prompt_tool_fn
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

def test_divider_and_prompt(research_output=None):
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
        else:
            print("First scene logic error: no description found.")
    else:
        print("Divider tool failed to generate scenes or output is not a list.")

if __name__ == "__main__":
    import sys
    
    print("Select test to run:")
    print("1. Research + Divider + Prompt (Full Flow)")
    print("2. Divider + Prompt (Fast, using mock data)")
    
    choice = input("Enter choice (1/2): ")
    
    if choice == '1':
        res = test_research()
        test_divider_and_prompt(res)
    elif choice == '2':
        test_divider_and_prompt()
    else:
        print("Invalid choice.")
