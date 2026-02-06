import os
import time
import json
import requests
from google.genai import types
from config import MODEL_NAME, SAM_API_URL
from . import utils

def segmentation_tool_fn(image_path: str) -> str:
    """
    Identifies main objects in an image using Gemini, then segments them using a hosted SAM3 model.
    
    Args:
        image_path: The path to the image file.
    Returns:
        The path to the saved JSON file containing segmentation results (masks, boxes, scores).
    """
    if not utils.client:
        return "Error: GEMINI_API_KEY not configured."
    
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
    
    print(f"Starting segmentation process for: {image_path}")

    # 1. Identify objects with Gemini
    prompt = 'Identify the main, distinct physical objects in this image that would be good candidates for instance segmentation. Return a raw JSON list of strings, for example: ["cat", "hat", "table"]. Do not include markdown formatting or explanation.'
    
    objects = []
    try:
        # Determine mime type
        mime_type = "image/png"
        if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
            mime_type = "image/jpeg"
            
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        contents = [
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        ]
        
        # Use the standard model for identification, assuming it has vision capabilities
        # config.py says MODEL_NAME = "gemini-2.5-flash", which is multimodal
        response = utils.client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config={
                'response_mime_type': 'application/json'
            }
        )
        
        objects = json.loads(response.text)
        print(f"Identified objects: {objects}")
        
    except Exception as e:
        return f"Error identifying objects with Gemini: {str(e)}"

    if not objects or not isinstance(objects, list):
         return "Error: Gemini failed to return a valid list of objects."

    # 2. Call SAM3 for each object
    api_url = SAM_API_URL
    combined_results = {
        "image_path": image_path,
        "objects": objects,
        "segmentations": {}
    }
    
    for obj in objects:
        print(f"Segmenting object: {obj}...")
        try:
           # Re-open file for each request to ensure pointer is at start
           with open(image_path, "rb") as f:
               files = {"file": f}
               data = {"prompt": obj} 
               
               response = requests.post(api_url, files=files, data=data) 
               
               if response.status_code == 200:
                   result = response.json()
                   combined_results["segmentations"][obj] = result
               else:
                   print(f"SAM3 failed for {obj}: {response.status_code} - {response.text}")
                   combined_results["segmentations"][obj] = {"error": f"Status {response.status_code}: {response.text}"}
                   
        except Exception as e:
            print(f"Error calling SAM3 for {obj}: {e}")
            combined_results["segmentations"][obj] = {"error": str(e)}

    # 3. Save combined results
    timestamp = int(time.time())
    output_filename = f"segmentation_results_{timestamp}.json"
    
    # Save to global output dir if set
    if utils.GLOBAL_OUTPUT_DIR:
        saved_path = utils._save_to_run_folder(json.dumps(combined_results, indent=2), output_filename)
        print(f"Segmentation results saved to: {saved_path}")
        return saved_path
    else:
        # Fallback: save to current directory
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(combined_results, indent=2))
            return os.path.abspath(output_filename)
        except:
            return json.dumps(combined_results, indent=2) # Return content as fallback
