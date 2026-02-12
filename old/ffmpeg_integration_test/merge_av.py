import subprocess
import json
import os
import argparse

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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def merge_av(video_path, audio_path, output_path):
    """Merge audio and video, freezing the last frame if audio is longer."""
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
        # We also need to ensure the audio is mapped and respects the longer duration
        # -shortest is NOT used because we want the longer audio
        filter_complex = f"[0:v]tpad=stop_mode=clone:stop_duration={pad_seconds}[v]"
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-pix_fmt", "yuv420p" # Ensure compatibility
        ])
    else:
        # Video is longer or equal, just map them
        # Usually, if video is longer, we might want the output to be video duration or audio duration.
        # The user said "else let video finish", which implies output should match video length.
        cmd.extend([
            "-map", "0:v",
            "-map", "1:a",
            "-pix_fmt", "yuv420p"
        ])
    
    cmd.append(output_path)
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error during ffmpeg execution: {result.stderr}")
    else:
        print(f"Successfully created: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Video and Audio with duration matching.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--audio", required=True, help="Path to input audio")
    parser.add_argument("--output", default="output.mp4", help="Path to output file")
    
    args = parser.parse_args()
    
    merge_av(args.video, args.audio, args.output)
