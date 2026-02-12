import os
import datetime
from tools import (
    research_tool_fn, 
    divider_tool_fn, 
    prompt_tool_fn, 
    image_gen_tool_fn, 
    generate_tts_audio_tool_fn,
    segmentation_tool_fn,
    merge_audio_video_tool_fn,
    concatenate_videos_tool_fn,
    burn_subtitles_to_video_tool_fn,
    transcribe_audio_tool_fn,
    draw_animation_tool_fn,
    set_output_dir
)

def run_pipeline(user_context: str, do_research: bool = True):
    print(f"--- Starting Storyboard Pipeline for context: {user_context} ---")
    
    # 0. Setup Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), "output", f"run_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    set_output_dir(output_dir)
    print(f"Artifacts will be saved to: {output_dir}")

    # 1. Research (Optional)
    research_report = user_context
    if do_research:
        print("\nStep 1: Performing Deep Research...")
        research_report = research_tool_fn(user_context)
        print("Research completed.")
    else:
        print("\nStep 1: Skipping Research as per request. Using provided context directly.")

    # 2. Divide into scenes
    print("\nStep 2: Dividing research/context into scenes...")
    scenes = divider_tool_fn(research_report)
    print(f"Generated {len(scenes)} scenes.")

    final_videos = []
    prev_image_path = None

    # 3. Asset Generation & Processing
    print("\nStep 3: Processing Scenes...")
    for i, scene in enumerate(scenes):
        print(f"\n--- Processing Scene {i+1}/{len(scenes)} ---")
        description = scene.get('description', 'No description')
        narration = scene.get('narration', 'No narration')
        
        # 3a. Generate Image Prompt
        print(f"Scene {i+1}: Generating image prompt...")
        img_prompt = prompt_tool_fn(description)
        
        # 3b. Generate Image
        # Pass the previous image as context for consistency if it exists
        print(f"Scene {i+1}: Generating image...")
        image_path = image_gen_tool_fn(img_prompt, reference_image_path=prev_image_path)
        
        if not image_path or "Error" in image_path:
            print(f"Error generating image for scene {i+1}: {image_path}")
            continue
            
        prev_image_path = image_path # Update for next scene
        
        # 3c. SAM Segmentation
        print(f"Scene {i+1}: Segmenting image objects...")
        seg_json_path = segmentation_tool_fn(image_path)
        
        # 3d. Whiteboard Animation Generation
        print(f"Scene {i+1}: Generating whiteboard animation...")
        anim_video_path = draw_animation_tool_fn(image_path, segmentation_results_path=seg_json_path)
        
        # 3e. TTS Generation
        print(f"Scene {i+1}: Generating narration audio...")
        audio_path = generate_tts_audio_tool_fn(narration)
        
        # 3f. Audio-Video Merging
        print(f"Scene {i+1}: Merging audio and video...")
        merged_output = os.path.join(output_dir, f"scene_{i+1}_merged.mp4")
        merged_video_path = merge_audio_video_tool_fn(anim_video_path, audio_path, merged_output)
        
        # 3g. Audio Transcription (for subtitles)
        print(f"Scene {i+1}: Transcribing audio for subtitles...")
        subtitles_json_path = transcribe_audio_tool_fn(merged_video_path)
        
        # 3h. Subtitle Burning
        if subtitles_json_path and os.path.exists(subtitles_json_path):
            print(f"Scene {i+1}: Burning subtitles into video...")
            subtitled_output = os.path.join(output_dir, f"scene_{i+1}_final.mp4")
            final_scene_video = burn_subtitles_to_video_tool_fn(merged_video_path, subtitles_json_path, subtitled_output)
            
            if final_scene_video and os.path.exists(final_scene_video):
                final_videos.append(final_scene_video)
            else:
                print(f"Warning: Subtitle burning failed for scene {i+1}: {final_scene_video}. Using merged video.")
                final_videos.append(merged_video_path)
        else:
            print(f"Warning: Transcription failed for scene {i+1}: {subtitles_json_path}. Skipping subtitles.")
            final_videos.append(merged_video_path)

    # 4. Final Merge
    if len(final_videos) >= 2:
        print("\nStep 4: Concatenating all scenes into a final video...")
        final_video_path = os.path.join(output_dir, "storyboard_final_video.mp4")
        result = concatenate_videos_tool_fn(final_videos, final_video_path)
        print(f"\n--- Pipeline Complete! ---")
        print(f"Final Video: {result}")
        return result
    elif len(final_videos) == 1:
        print(f"\n--- Pipeline Complete (Single Scene)! ---")
        print(f"Final Video: {final_videos[0]}")
        return final_videos[0]
    else:
        print("\nPipeline failed: No videos generated.")
        return None

if __name__ == "__main__":
    context = input("Enter the context for your video: ")
    research_choice = input("Run Deep Research? (y/n): ").lower() == 'y'
    run_pipeline(context, do_research=research_choice)
