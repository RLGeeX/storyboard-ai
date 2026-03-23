from tools import research_tool, director_tool_fn as director_tool, prompt_tool

def run_storyboard_agent(user_context: str):
    print(f"--- Starting Storyboard Agent for context: {user_context} ---")
    
    # Step 1: Research
    print("Step 1: Performing Deep Research...")
    research_report = research_tool(user_context)
    print("Research completed.")
    
    # Step 2: Director plans the video
    print("Step 2: Director planning scenes & narration...")
    scenes_raw = director_tool(user_context)
    print("Scenes generated.")
    
    # Step 3: Generate prompts for each scene
    print("Step 3: Generating image prompts for each scene...")
    storyboard = []
    for i, scene in enumerate(scenes_raw):
        print(f"Processing scene {i+1}/{len(scenes_raw)}...")
        img_prompt = prompt_tool(scene['description'])
        storyboard.append({
            "scene_number": i + 1,
            "description": scene['description'],
            "narration": scene['narration'],
            "image_prompt": img_prompt
        })
    
    print("--- Storyboard Generation Complete ---")
    for item in storyboard:
        print(f"\nScene {item['scene_number']}:")
        print(f"Description: {item['description']}")
        print(f"Narration: {item['narration']}")
        print(f"Prompt: {item['image_prompt']}")

if __name__ == "__main__":
    context = input("Enter the context for your storyboard video: ")
    run_storyboard_agent(context)
