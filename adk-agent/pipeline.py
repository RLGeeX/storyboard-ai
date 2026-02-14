import os
import time
import datetime
from tools import (
    research_tool_fn, 
    director_tool_fn, 
    prompt_tool_fn, 
    image_gen_tool_fn, 
    generate_tts_audio_tool_fn,
    segmentation_tool_fn,
    merge_audio_video_tool_fn,
    concatenate_videos_tool_fn,
    burn_subtitles_to_video_tool_fn,
    transcribe_audio_tool_fn,
    refine_narration_tool_fn,
    draw_animation_tool_fn,
    set_output_dir,
    get_video_duration
)

# --- Helper functions for robustness ---

def _is_valid_path(path: str) -> bool:
    """Check if a tool returned a valid file path (not an error string)."""
    if not path:
        return False
    if "Error" in path or "error" in path:
        return False
    return os.path.exists(path)

def _retry(fn, *args, max_retries: int = 3, delay: float = 5.0, label: str = "", **kwargs):
    """
    Retry a function call on failure (handles transient network errors).
    Returns the result on success, or None on exhausted retries.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            # Check if result is an error string (some tools return error strings instead of raising)
            if isinstance(result, str) and ("Error" in result or "error" in result) and not os.path.exists(result):
                raise RuntimeError(result)
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"  ⚠ {label} failed (attempt {attempt}/{max_retries}): {e}")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  ✗ {label} failed after {max_retries} attempts: {e}")
    return None

# --- Main Pipeline ---

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
    research_report = None
    if do_research:
        print("\nStep 1: Performing Deep Research...")
        research_report = research_tool_fn(user_context)
        print("Research completed.")
    else:
        print("\nStep 1: Skipping Research as per request. Using provided context directly.")

    # Step 2: Director Planning — The Director plans the entire video journey
    print("\nStep 2: Director Planning & Scene Writing...")
    video_plan = director_tool_fn(user_context, research_material=research_report)
    global_plan = video_plan.get("global_plan", {})
    scenes = video_plan.get("scenes", [])
    print(f"Director planned {len(scenes)} scenes. Tone: {global_plan.get('tone')}, Arc: {global_plan.get('narrative_arc', 'N/A')}")

    final_videos = []
    prev_image_path = None
    failed_scenes = []

    # 3. Asset Generation & Processing
    print("\nStep 3: Processing Scenes...")
    for i, scene in enumerate(scenes):
        scene_num = i + 1
        print(f"\n{'='*60}")
        print(f"--- Processing Scene {scene_num}/{len(scenes)} ---")
        
        try:
            description = scene.get('description', 'No description')
            narration = scene.get('narration', 'No narration')
            visual_setup = scene.get('visual_setup', '')
            summary = scene.get('summary', '')
            emotional_beat = scene.get('emotional_beat', '')
            
            if summary:
                print(f"  Summary: {summary}")
            if emotional_beat:
                print(f"  Emotional Beat: {emotional_beat}")
            
            # --- 3a. Generate Image Prompt (with retry) ---
            print(f"Scene {scene_num}: Generating image prompt...")
            img_prompt = _retry(
                prompt_tool_fn, description, 
                visual_setup=visual_setup, global_plan=global_plan,
                label=f"Scene {scene_num} image prompt", max_retries=2
            )
            if not img_prompt:
                print(f"  ✗ SKIPPING Scene {scene_num}: Image prompt generation failed.")
                failed_scenes.append(scene_num)
                continue
            
            # --- 3b. Generate Image (with retry) ---
            print(f"Scene {scene_num}: Generating image...")
            image_path = _retry(
                image_gen_tool_fn, img_prompt,
                reference_image_path=prev_image_path,
                label=f"Scene {scene_num} image gen", max_retries=3, delay=8.0
            )
            
            if not _is_valid_path(image_path):
                print(f"  ✗ SKIPPING Scene {scene_num}: Image generation failed — no valid image produced.")
                failed_scenes.append(scene_num)
                continue
                
            prev_image_path = image_path 

            # --- 3c. SAM Segmentation (non-critical, can fail gracefully) ---
            print(f"Scene {scene_num}: Segmenting image objects...")
            seg_json_path = None
            try:
                seg_json_path = segmentation_tool_fn(image_path)
                if not _is_valid_path(seg_json_path):
                    print(f"  ⚠ Segmentation returned no valid result. Continuing without segmentation.")
                    seg_json_path = None
            except Exception as e:
                print(f"  ⚠ Segmentation failed (non-critical): {e}. Continuing without segmentation.")
            
            # --- 3d. Whiteboard Animation Generation ---
            print(f"Scene {scene_num}: Generating whiteboard animation...")
            anim_video_path = draw_animation_tool_fn(image_path, segmentation_results_path=seg_json_path)
            
            if not _is_valid_path(anim_video_path):
                print(f"  ✗ SKIPPING Scene {scene_num}: Animation generation failed.")
                failed_scenes.append(scene_num)
                continue
            
            # --- 3e. Narration Refinement (non-critical, can fallback to original) ---
            v_duration = get_video_duration(anim_video_path)
            print(f"Scene {scene_num}: Refining narration (Duration: {v_duration:.1f}s)...")
            try:
                refined = refine_narration_tool_fn(narration, image_path, video_duration=v_duration, global_plan=global_plan)
                if refined and "Error" not in refined:
                    narration = refined
                else:
                    print(f"  ⚠ Narration refinement returned error. Using Director's original narration.")
            except Exception as e:
                print(f"  ⚠ Narration refinement failed (non-critical): {e}. Using Director's original narration.")
            
            # --- 3f. TTS Generation (with retry) ---
            print(f"Scene {scene_num}: Generating narration audio...")
            audio_path = _retry(
                generate_tts_audio_tool_fn, narration,
                label=f"Scene {scene_num} TTS", max_retries=3, delay=5.0
            )
            
            if not _is_valid_path(audio_path):
                print(f"  ✗ SKIPPING Scene {scene_num}: TTS generation failed — no audio produced.")
                failed_scenes.append(scene_num)
                continue
            
            # --- 3g. Audio-Video Merging ---
            print(f"Scene {scene_num}: Merging audio and video...")
            merged_output = os.path.join(output_dir, f"scene_{scene_num}_merged.mp4")
            merged_video_path = merge_audio_video_tool_fn(anim_video_path, audio_path, merged_output)
            
            if not _is_valid_path(merged_video_path):
                print(f"  ✗ SKIPPING Scene {scene_num}: Audio-Video merge failed.")
                failed_scenes.append(scene_num)
                continue
            
            # --- 3h. Audio Transcription (non-critical) ---
            print(f"Scene {scene_num}: Transcribing audio for subtitles...")
            subtitles_json_path = None
            try:
                subtitles_json_path = transcribe_audio_tool_fn(merged_video_path)
            except Exception as e:
                print(f"  ⚠ Transcription failed (non-critical): {e}. Skipping subtitles.")
            
            # --- 3i. Subtitle Burning (non-critical) ---
            if _is_valid_path(subtitles_json_path):
                print(f"Scene {scene_num}: Burning subtitles into video...")
                subtitled_output = os.path.join(output_dir, f"scene_{scene_num}_final.mp4")
                try:
                    final_scene_video = burn_subtitles_to_video_tool_fn(merged_video_path, subtitles_json_path, subtitled_output)
                    
                    if _is_valid_path(final_scene_video):
                        final_videos.append(final_scene_video)
                    else:
                        print(f"  ⚠ Subtitle burning failed. Using merged video without subtitles.")
                        final_videos.append(merged_video_path)
                except Exception as e:
                    print(f"  ⚠ Subtitle burning error: {e}. Using merged video without subtitles.")
                    final_videos.append(merged_video_path)
            else:
                print(f"  ⚠ No subtitles available. Using merged video as-is.")
                final_videos.append(merged_video_path)
            
            print(f"  ✓ Scene {scene_num} completed successfully!")
                
        except Exception as e:
            print(f"  ✗ UNEXPECTED ERROR in Scene {scene_num}: {e}")
            failed_scenes.append(scene_num)
            continue

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Scene Processing Summary: {len(final_videos)} succeeded, {len(failed_scenes)} failed")
    if failed_scenes:
        print(f"Failed scenes: {failed_scenes}")

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


