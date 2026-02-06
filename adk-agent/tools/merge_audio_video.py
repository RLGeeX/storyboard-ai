import subprocess
import json
import os

def get_duration(file_path):
    """Get duration of a file in seconds using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "quiet", 
        "-print_format", "json", 
        "-show_format", 
        "-show_streams", 
        file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except (KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to parse ffprobe output for {file_path}: {e}")

def merge_audio_video_tool_fn(video_path: str, audio_path: str, output_path: str = "output.mp4") -> str:
    """
    Merges audio and video files. If audio is longer than video, the last frame of the video 
    is frozen to match the audio duration.
    
    Args:
        video_path: Path to the input video file.
        audio_path: Path to the input audio file.
        output_path: Path for the merged output file.
        
    Returns:
        Path to the output file if successful, or error message.
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"
    if not os.path.exists(audio_path):
        return f"Error: Audio file not found at {audio_path}"
        
    try:
        video_dur = get_duration(video_path)
        audio_dur = get_duration(audio_path)
        
        print(f"Video duration: {video_dur:.2f}s")
        print(f"Audio duration: {audio_dur:.2f}s")
        
        cmd = [
            "ffmpeg",
            "-y", # Overwrite output if exists
            "-i", video_path,
            "-i", audio_path,
        ]
        
        if audio_dur > video_dur:
            # Pad the video at the end by cloning the last frame
            pad_seconds = audio_dur - video_dur
            # Using tpad filter to clone the last frame
            filter_complex = f"[0:v]tpad=stop_mode=clone:stop_duration={pad_seconds}[v]"
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "1:a",
                "-pix_fmt", "yuv420p" # Ensure compatibility
            ])
        else:
            # Video is longer or equal, just map them
            cmd.extend([
                "-map", "0:v",
                "-map", "1:a",
                "-pix_fmt", "yuv420p"
            ])
        
        cmd.append(output_path)
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return f"Error during ffmpeg execution: {result.stderr}"
        else:
            print(f"Successfully created: {output_path}")
            return output_path
            
    except Exception as e:
        return f"Error in merge_audio_video_tool: {str(e)}"
