import time
from config import DEEP_RESEARCH_MODEL, MODEL_NAME
from .utils import client, _save_to_run_folder

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
                    report = interaction.outputs[-1].text
                    _save_to_run_folder(report, "research_report.md")
                    return report
                return "Research completed with no output."
            elif interaction.status == "failed":
                return f"Research failed: {interaction.error}"
            
            time.sleep(10)

    except Exception as e:
        return f"An error occurred during research: {str(e)}"

def web_grounded_research_tool_fn(context: str) -> str:
    """
    Performs fast, web-grounded research using standard Gemini model with Google Search Tool enabled.
    
    Args:
        context: The topic to research.
    Returns:
        A concise, factual summary.
    """
    print(f"Starting Web-Grounded Research for: {context}")
    
    if not client:
        return "Error: GEMINI_API_KEY not configured."

    try:
        from google.genai import types
        
        prompt = f"Perform a comprehensive web search to provide a detailed, factual summary about: {context}. Include key dates, milestones, and important contextual facts. This will be used as a source for a documentary/storyboard script."
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
            )
        )
        
        report = response.text.strip()
        _save_to_run_folder(report, "web_research_report.md")
        return report

    except Exception as e:
        return f"An error occurred during web-grounded research: {str(e)}"
