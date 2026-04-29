import os
import sys
import time
import datetime
from tools import (
    research_tool_fn,
    web_grounded_research_tool_fn,
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
    get_video_duration,
    reference_search_tool_fn
)
from tools.script_parser import parse_script_file

# --- Tee logger: write all stdout/stderr to a log file alongside the run output ---

class _Tee:
    """Duplicate writes to multiple streams (e.g., console + log file)."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass
    def isatty(self):
        return False


def _install_log_tee(log_path: str):
    """Tee stdout and stderr into log_path. Console output is preserved."""
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_file


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

def run_pipeline(
    user_context: str,
    do_research: bool = True,
    do_web_search: bool = False,
    use_internet_image_search: bool = True,
    prebuilt_plan: dict = None,
):
    """Execute the storyboard pipeline.

    When `prebuilt_plan` is provided (e.g., from a parsed script file), the
    research and Director steps are skipped entirely. The plan is used as-is,
    so each scene's narration is whatever the author wrote — no LLM rewrites.
    """
    # 0. Setup Output Directory + tee log
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), "output", f"run_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    set_output_dir(output_dir)

    log_path = os.path.join(output_dir, "pipeline.log")
    _install_log_tee(log_path)
    print(f"Logging to: {log_path}")

    if prebuilt_plan:
        # ----- Script-file path: skip research + Director, use plan verbatim -----
        global_plan = prebuilt_plan.get("global_plan", {})
        scenes = prebuilt_plan.get("scenes", [])
        title = global_plan.get("title", "(untitled)")
        print(f"--- Starting Storyboard Pipeline from prebuilt plan: {title} ---")
        print(f"Artifacts will be saved to: {output_dir}")
        print("\nStep 1: Skipping research (prebuilt plan provided).")
        print("\nStep 2: Skipping Director (prebuilt plan provided).")
        print(f"Plan has {len(scenes)} scenes. Source: {global_plan.get('source', 'unknown')}")
        if global_plan.get("style_preamble"):
            preview = global_plan["style_preamble"][:120]
            print(f"Style preamble in effect: {preview}{'...' if len(global_plan['style_preamble']) > 120 else ''}")
        # Persist the plan so the run directory has the same artifact the Director path produces.
        try:
            import json as _json
            with open(os.path.join(output_dir, "video_plan.json"), "w", encoding="utf-8") as f:
                _json.dump(prebuilt_plan, f, indent=2)
        except Exception as e:
            print(f"  (note: could not persist video_plan.json: {e})")
    else:
        # ----- Default path: research + Director plan the video -----
        print(f"--- Starting Storyboard Pipeline for context: {user_context} ---")
        print(f"Artifacts will be saved to: {output_dir}")

        # 1. Research (Optional)
        research_report = None
        if do_research:
            print("\nStep 1: Performing Deep Research...")
            research_report = research_tool_fn(user_context)
            print("Research completed.")
        elif do_web_search:
            print("\nStep 1: Performing Web-Grounded Research (Fast)...")
            research_report = web_grounded_research_tool_fn(user_context)
            print("Web-Grounded Research completed.")
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
            search_query = scene.get('search_query', '')
            text_overlay = scene.get('text_overlay', '')
            
            if summary:
                print(f"  Summary: {summary}")
            if emotional_beat:
                print(f"  Emotional Beat: {emotional_beat}")
                
            # --- 3.a.0 Reference Search ---
            subject_image_path = None
            if use_internet_image_search and search_query:
                print(f"Scene {scene_num}: Searching internet for reference image: '{search_query}'...")
                res = reference_search_tool_fn(search_query)
                if _is_valid_path(res):
                    subject_image_path = res
                    print(f"  ✓ Reference image downloaded to: {subject_image_path}")
                else:
                    print(f"  ⚠ Reference search failed or returned no valid image: {res}")
            elif not use_internet_image_search and search_query:
                print(f"Scene {scene_num}: Internet image search disabled. Skipping reference for '{search_query}'.")
            
            # --- 3a. Generate Image Prompt (with retry) ---
            print(f"Scene {scene_num}: Generating image prompt...")
            img_prompt = _retry(
                prompt_tool_fn, description, 
                visual_setup=visual_setup, text_overlay=text_overlay, global_plan=global_plan,
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
                subject_reference_image_path=subject_image_path,
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
            # Skip refinement when the plan came from a script file — the author
            # supplied the narration verbatim and doesn't want LLM rewrites.
            v_duration = get_video_duration(anim_video_path)
            if global_plan.get("source") == "script_file":
                print(f"Scene {scene_num}: Skipping narration refinement (script-file mode — narration is verbatim).")
            else:
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
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate a whiteboard explainer video from a prompt or script file.",
    )
    parser.add_argument(
        "--script-file", "-s",
        type=str,
        help=(
            "Path to a structured markdown script file with '## Scene N' headers "
            "and **Narration**/**On-screen text**/**Visual cue** blocks. "
            "Bypasses the LLM Director — narration is used VERBATIM. "
            "The script's **Style:** frontmatter is propagated to every image prompt."
        ),
    )
    parser.add_argument(
        "--prompt-file", "-f",
        type=str,
        help="Path to a file containing the video context/script. Avoids terminal-paste issues with multi-line markdown.",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Inline prompt text. Use --prompt-file for multi-line input.",
    )
    parser.add_argument(
        "--research", "-r",
        choices=["deep", "web", "none"],
        default="web",
        help="Research mode: 'deep' (Deep Research), 'web' (Web Search, fast - default), 'none'.",
    )
    parser.add_argument(
        "--no-image-search",
        action="store_true",
        help="Disable internet image search for visual reference grounding.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Force interactive prompts even when other args are provided.",
    )
    args = parser.parse_args()

    # If --script-file is provided, parse it and run the pipeline with the
    # prebuilt plan. Skips the LLM Director and uses narration verbatim.
    if args.script_file:
        script_path = os.path.expanduser(args.script_file)
        if not os.path.exists(script_path):
            print(f"ERROR: script file not found: {script_path}")
            sys.exit(1)
        try:
            video_plan = parse_script_file(script_path)
        except Exception as e:
            print(f"ERROR: failed to parse {script_path}: {e}")
            sys.exit(1)
        print(
            f"Loaded {len(video_plan.get('scenes', []))} scenes from {script_path}"
        )
        run_pipeline(
            user_context=video_plan["global_plan"].get("title", "(from script file)"),
            do_research=False,
            do_web_search=False,
            use_internet_image_search=not args.no_image_search,
            prebuilt_plan=video_plan,
        )
        sys.exit(0)

    # Resolve context
    context = None
    if args.prompt_file:
        prompt_path = os.path.expanduser(args.prompt_file)
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                context = f.read().strip()
            print(f"Loaded prompt from {prompt_path} ({len(context)} chars)")
        except FileNotFoundError:
            print(f"ERROR: prompt file not found: {prompt_path}")
            sys.exit(1)
    elif args.prompt:
        context = args.prompt
    elif args.interactive or sys.stdin.isatty():
        # Interactive: ask for a file path instead of inline text. This avoids the
        # terminal-paste problem with multi-line markdown (newlines would submit early).
        while True:
            raw = input("Enter path to prompt file (or '-' for stdin): ").strip()
            if raw == "-":
                print("Reading prompt from stdin (end with Ctrl-D):")
                context = sys.stdin.read().strip()
                break
            # Strip surrounding quotes if user pastes a quoted path
            if len(raw) >= 2 and raw[0] in ("'", '"') and raw[-1] == raw[0]:
                raw = raw[1:-1]
            prompt_path = os.path.expanduser(raw)
            if os.path.exists(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        context = f.read().strip()
                    print(f"Loaded prompt from {prompt_path} ({len(context)} chars)")
                    break
                except Exception as e:
                    print(f"  Could not read file: {e}")
            else:
                print(f"  File not found: {prompt_path}")
                retry = input("  Try another path? [Y/n]: ").strip().lower()
                if retry == "n":
                    sys.exit(1)
    else:
        context = sys.stdin.read().strip()

    if not context:
        print("ERROR: empty context. Use --prompt-file <path> or --prompt '<text>'.")
        sys.exit(1)

    # Resolve research mode (CLI flag wins; otherwise interactive default if no flags given)
    if args.interactive and not (args.prompt_file or args.prompt):
        res_choice = input("Select research mode: [1] Deep Research, [2] Web Search (Fast), [3] None (default 2): ").strip()
        if res_choice == '1':
            do_research, do_web_search = True, False
        elif res_choice == '3':
            do_research, do_web_search = False, False
        else:
            do_research, do_web_search = False, True
        image_search_choice = input("Enable internet image search for references? [Y/n] (default Y): ").strip().lower()
        use_internet_image_search = False if image_search_choice in ['n', 'no'] else True
    else:
        do_research = (args.research == "deep")
        do_web_search = (args.research == "web")
        use_internet_image_search = not args.no_image_search

    run_pipeline(
        context,
        do_research=do_research,
        do_web_search=do_web_search,
        use_internet_image_search=use_internet_image_search,
    )


