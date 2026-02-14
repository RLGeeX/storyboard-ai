"""
Test script to compare Pillow vs FFmpeg for subtitle rendering.
Demonstrates:
1. High-quality text rendering with anti-aliasing
2. Automatic word wrapping
3. Semi-transparent background for readability
"""

import os
import subprocess
import textwrap
from PIL import Image, ImageDraw, ImageFont

# Configuration
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
SUBTITLE_TEXT = "This is a sample subtitle text that is quite long and should automatically wrap to multiple lines when it exceeds the available width of the video frame."

# Try to use a nice font, fallback to default
def get_font(size=36):
    """Get a TrueType font, with fallback options."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def create_test_image():
    """Create a sample test image (gradient background to show text quality)."""
    img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT))
    draw = ImageDraw.Draw(img)
    
    # Create a gradient background to better show text quality
    for y in range(IMAGE_HEIGHT):
        # Blue to purple gradient
        r = int(30 + (y / IMAGE_HEIGHT) * 50)
        g = int(50 + (y / IMAGE_HEIGHT) * 30)
        b = int(100 + (y / IMAGE_HEIGHT) * 80)
        draw.line([(0, y), (IMAGE_WIDTH, y)], fill=(r, g, b))
    
    # Add some sample content
    font = get_font(48)
    draw.text((IMAGE_WIDTH//2, IMAGE_HEIGHT//3), "Sample Video Frame", 
              fill=(255, 255, 255), font=font, anchor="mm")
    
    test_image_path = os.path.join(OUTPUT_DIR, "test_frame.png")
    img.save(test_image_path)
    print(f"Created test image: {test_image_path}")
    return test_image_path


def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def method_pillow(input_image_path, subtitle_text, output_path):
    """
    Method 1: Pillow (PIL) - High quality text with word wrapping.
    Creates a transparent overlay with text and composites it.
    """
    print("\n=== Method 1: Pillow ===")
    
    # Load the background image
    background = Image.open(input_image_path).convert('RGBA')
    width, height = background.size
    
    # Create transparent overlay for subtitle
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Font settings
    font = get_font(36)
    padding = 20
    max_text_width = width - (padding * 4)
    
    # Wrap text to fit width
    lines = wrap_text(subtitle_text, font, max_text_width, draw)
    
    # Calculate text block dimensions
    line_height = 45
    total_text_height = len(lines) * line_height
    
    # Position at bottom of frame
    box_y = height - total_text_height - (padding * 3)
    box_height = total_text_height + (padding * 2)
    
    # Draw semi-transparent background box
    draw.rectangle(
        [(padding, box_y), (width - padding, box_y + box_height)],
        fill=(0, 0, 0, 180)  # Semi-transparent black
    )
    
    # Draw each line of text
    y_offset = box_y + padding
    for line in lines:
        # Get text width for centering
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x_pos = (width - text_width) // 2
        
        # Draw text with slight shadow for better readability
        draw.text((x_pos + 2, y_offset + 2), line, fill=(0, 0, 0, 200), font=font)  # Shadow
        draw.text((x_pos, y_offset), line, fill=(255, 255, 255, 255), font=font)    # Main text
        y_offset += line_height
    
    # Composite overlay onto background
    result = Image.alpha_composite(background, overlay)
    result = result.convert('RGB')
    result.save(output_path, quality=95)
    
    print(f"Pillow output saved: {output_path}")
    return output_path


def method_ffmpeg(input_image_path, subtitle_text, output_path):
    """
    Method 2: FFmpeg drawtext filter - Direct text burning.
    Uses FFmpeg's built-in text rendering with word wrap.
    """
    print("\n=== Method 2: FFmpeg drawtext ===")
    
    # Escape special characters for FFmpeg
    escaped_text = subtitle_text.replace("'", "'\\''").replace(":", "\\:")
    
    # FFmpeg drawtext with word wrapping simulation
    # Note: FFmpeg doesn't have native word wrap, so we use line_spacing trick
    # For proper wrapping, we preprocess with textwrap
    wrapped_lines = textwrap.wrap(subtitle_text, width=60)
    multiline_text = '\n'.join(wrapped_lines)
    escaped_text = multiline_text.replace("'", "'\\''").replace(":", "\\:")
    
    # Build FFmpeg command with drawtext filter
    # Using a box behind text for readability
    cmd = [
        "ffmpeg", "-y",
        "-i", input_image_path,
        "-vf", (
            f"drawtext="
            f"text='{escaped_text}':"
            f"fontfile='C\\:/Windows/Fonts/arial.ttf':"
            f"fontsize=36:"
            f"fontcolor=white:"
            f"borderw=2:"
            f"bordercolor=black:"
            f"x=(w-text_w)/2:"
            f"y=h-text_h-40:"
            f"box=1:"
            f"boxcolor=black@0.7:"
            f"boxborderw=15"
        ),
        output_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"FFmpeg output saved: {output_path}")
            return output_path
        else:
            print(f"FFmpeg error: {result.stderr}")
            return None
    except Exception as e:
        print(f"FFmpeg failed: {e}")
        return None


def method_opencv_comparison(input_image_path, subtitle_text, output_path):
    """
    Method 3: OpenCV (for comparison) - Shows the poor quality.
    """
    print("\n=== Method 3: OpenCV (for comparison) ===")
    
    try:
        import cv2
        import numpy as np
        
        img = cv2.imread(input_image_path)
        height, width = img.shape[:2]
        
        # OpenCV putText - no word wrap, basic font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Truncate text since OpenCV won't wrap
        display_text = subtitle_text[:80] + "..." if len(subtitle_text) > 80 else subtitle_text
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
        
        # Position at bottom
        x = (width - text_width) // 2
        y = height - 40
        
        # Draw background rectangle
        cv2.rectangle(img, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img, display_text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        cv2.imwrite(output_path, img)
        print(f"OpenCV output saved: {output_path}")
        print("Note: Text is truncated because OpenCV doesn't support word wrapping!")
        return output_path
    except ImportError:
        print("OpenCV not installed, skipping comparison")
        return None


def main():
    print("=" * 60)
    print("Subtitle Rendering Methods Comparison")
    print("=" * 60)
    
    # Create test image
    test_image = create_test_image()
    
    # Test all methods
    pillow_output = method_pillow(
        test_image, 
        SUBTITLE_TEXT,
        os.path.join(OUTPUT_DIR, "output_pillow.png")
    )
    
    ffmpeg_output = method_ffmpeg(
        test_image,
        SUBTITLE_TEXT, 
        os.path.join(OUTPUT_DIR, "output_ffmpeg.png")
    )
    
    opencv_output = method_opencv_comparison(
        test_image,
        SUBTITLE_TEXT,
        os.path.join(OUTPUT_DIR, "output_opencv.png")
    )
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. Pillow:  {pillow_output}")
    print(f"  2. FFmpeg:  {ffmpeg_output}")
    print(f"  3. OpenCV:  {opencv_output}")
    print("\nCompare the outputs to see the quality difference!")


if __name__ == "__main__":
    main()
