import asyncio
import os
from google.adk import Agent, Runner
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from config import MODEL_NAME, GEMINI_API_KEY
from tools import research_tool_fn, director_tool_fn, prompt_tool_fn, image_gen_tool_fn, generate_tts_audio_tool_fn, set_output_dir
import time

# Ensure GOOGLE_API_KEY is set for ADK
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Define Tools
research_tool = FunctionTool(research_tool_fn)
director_tool = FunctionTool(director_tool_fn)
prompt_tool = FunctionTool(prompt_tool_fn)
image_gen_tool = FunctionTool(image_gen_tool_fn)
tts_tool = FunctionTool(generate_tts_audio_tool_fn)

async def main():
    print("--- Initializing Storyboard ADK Agent ---")
    
    # Create a fresh output directory for this run
    timestamp = int(time.time())
    run_output_dir = os.path.join("outputs", f"run_{timestamp}")
    set_output_dir(run_output_dir)
    
    # Configure the Model
    model = Gemini(model=MODEL_NAME)
    
    # Define the Agent
    agent = Agent(
        name="storyboard_agent",
        model=model,
        tools=[research_tool, director_tool, prompt_tool, image_gen_tool, tts_tool],
        instruction="""
        You are an autonomous Storyboard Director Agent. 
        Your goal is to create a detailed storyboard plan, image prompts, and generate the final images for a video.

        CRITICAL INSTRUCTION:
        1. Analyze the user's input topic.
        2. IF the input contains sufficient detail to create scenes (e.g., a full story or detailed description), SKIP research and proceed directly to dividing the scenes.
        3. IF the input is vague or just a topic name (e.g., "History of Chess"), call `research_tool_fn` to get detailed information FIRST.
        
        Workflow:
        1. (Optional) Call `research_tool_fn` if more context is needed.
        2. Call `director_tool_fn` (with either research output or original context) to get a list of scenes. 
        3. For EACH scene returned by `director_tool_fn`:
           - Call `prompt_tool_fn` with the scene description to generate a specialized whiteboard image prompt.
           - Call `image_gen_tool_fn` with the generated prompt to create the visual asset. 
           - For aesthetic consistency, pass the path of the previously generated image as `reference_image_path` to the NEXT `image_gen_tool_fn` call if it exists.
           - Call `generate_tts_audio_tool_fn` with the scene's narration to generate high-quality audio for that scene.
        4. Compile all results into a final storyboard format, including paths to the generated images and audio files.
        5. Output the final storyboard plan with scenes (narration, description), their corresponding image prompts, image file paths, and audio file paths.
        """
    )
    
    # Session Management
    APP_NAME = "storyboard_app"
    USER_ID = "current_user"
    SESSION_ID = "session_001"
    
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # Runner Setup
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    context = input("Enter the context for your storyboard video: ")
    print(f"\n--- Agent starting work on: {context} ---")
    
    # Save the input context to the run folder
    context_file_path = os.path.join(run_output_dir, "input_context.txt")
    with open(context_file_path, "w", encoding="utf-8") as f:
        f.write(context)
    
    # Create the new message object as expected by run_async
    new_message = types.Content(
        role="user",
        parts=[types.Part(text=f"Create a storyboard for: {context}")]
    )

    print("Agent is thinking and using tools... (This may take a while)\n")
    
    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=new_message
    ):
        # Handle different event types if needed
        # For now, we just want to see the progress and final output
        if hasattr(event, 'text') and event.text:
            print(event.text, end="", flush=True)
            final_text += event.text
        elif hasattr(event, 'call_stack_event'):
            # Tool calls or sub-agent calls
            # print(f"\n[Agent Action: {event.call_stack_event}]")
            pass

    print("\n\n--- Agent Work Complete ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
