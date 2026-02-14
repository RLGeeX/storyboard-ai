import asyncio
import os
from google.adk import Agent, Runner
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from config import MODEL_NAME, GEMINI_API_KEY
from tools import research_tool_fn, director_tool_fn, prompt_tool_fn, image_gen_tool_fn

# Ensure GOOGLE_API_KEY is set for ADK
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Define Tools
research_tool = FunctionTool(research_tool_fn)
director_tool = FunctionTool(director_tool_fn)
prompt_tool = FunctionTool(prompt_tool_fn)
image_gen_tool = FunctionTool(image_gen_tool_fn)

async def test_agent_workflow():
    print("--- Testing Storyboard ADK Agent Workflow ---")
    
    # Configure the Model
    model = Gemini(model=MODEL_NAME)
    
    # Define the Agent
    agent = Agent(
        name="storyboard_agent_test",
        model=model,
        tools=[research_tool, director_tool, prompt_tool, image_gen_tool],
        instruction="""
        You are an autonomous Storyboard Director Agent. 
        Your goal is to create a 2-scene storyboard and generate images for them.
        Use mock data or keep it very brief to save time.
        
        Workflow:
        1. Call `director_tool_fn` with a short story: "A cat finds a box. The cat enters the box."
        2. For EACH of the 2 scenes:
           - Call `prompt_tool_fn` for the description.
           - Call `image_gen_tool_fn` with the prompt.
        3. Output the final plan.
        """
    )
    
    # Session Management
    APP_NAME = "storyboard_test_app"
    USER_ID = "test_user"
    SESSION_ID = "test_session_001"
    
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
    
    print("\n--- Agent starting test run ---")
    
    new_message = types.Content(
        role="user",
        parts=[types.Part(text="Create a short 2-scene storyboard about a cat and a box.")]
    )

    print("Agent is thinking and using tools...\n")
    
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=new_message
    ):
        if hasattr(event, 'text') and event.text:
            print(event.text, end="", flush=True)

    print("\n\n--- Agent Test Work Complete ---")

if __name__ == "__main__":
    try:
        asyncio.run(test_agent_workflow())
    except Exception as e:
        print(f"\nAn error occurred: {e}")
