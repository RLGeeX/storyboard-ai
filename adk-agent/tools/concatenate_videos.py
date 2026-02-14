import subprocess
import os


def concatenate_videos_tool_fn(video_paths: list, output_path: str = "concatenated_output.mp4") -> str:
    """
    Concatenates multiple video files (with audio) sequentially into a single video.
    
    Args:
        video_paths: A list of paths to video files to concatenate. Must have at least 2 videos.
        output_path: Path for the concatenated output file.
        
    Returns:
        Path to the output file if successful, or an error message.
    """
    if not isinstance(video_paths, list) or len(video_paths) < 2:
        return "Error: video_paths must be a list with at least 2 video file paths."
    
    for i, vp in enumerate(video_paths):
        if not os.path.exists(vp):
            return f"Error: Video file not found at index {i}: {vp}"
    
    n = len(video_paths)
    
    # Build ffmpeg command with filter_complex for concat
    # Example for 2 videos:
    # ffmpeg -i video1.mp4 -i video2.mp4 -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]" -map "[v]" -map "[a]" output.mp4
    
    cmd = ["ffmpeg", "-y"]
    
    # Add inputs
    for vp in video_paths:
        cmd.extend(["-i", vp])
    
    # Build filter_complex string
    filter_inputs = "".join([f"[{i}:v][{i}:a]" for i in range(n)])
    filter_complex = f"{filter_inputs}concat=n={n}:v=1:a=1[v][a]"
    
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        "-pix_fmt", "yuv420p",
        output_path
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return f"Error during ffmpeg execution: {result.stderr}"
        else:
            print(f"Successfully concatenated {n} videos to: {output_path}")
            return output_path
    except Exception as e:
        return f"Error in concatenate_videos_tool: {str(e)}"
