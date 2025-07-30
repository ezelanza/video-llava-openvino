# Complete standalone webcam capture for LLaVA model
# Run this file directly: python webcam_capture_standalone.py

import cv2
import numpy as np
from pathlib import Path
import time
from transformers import LlavaNextVideoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

def capture_webcam_frames(duration_seconds=3, fps=4, width=320, height=240):
    """Capture frames from webcam for a specified duration automatically."""
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return []
    
    # Set resolution to reduce processing time
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Verify the resolution was set
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Webcam resolution set to: {actual_width}x{actual_height}")
    
    frames = []
    total_frames = duration_seconds * fps
    frame_interval = 1.0 / fps  # Time between frames
    
    print(f"Capturing {total_frames} frames over {duration_seconds} seconds...")
    print("Starting in 3 seconds...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Starting frame capture...")
    start_time = time.time()
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            # Resize frame to ensure consistent resolution
            frame_resized = cv2.resize(frame, (width, height))
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            elapsed_time = time.time() - start_time
            print(f"Captured frame {i+1}/{total_frames} at {elapsed_time:.1f}s")
            
            # Wait for next frame time
            if i < total_frames - 1:  # Don't wait after the last frame
                time.sleep(frame_interval)
    
    cap.release()
    print("Capture complete!")
    return frames

def main():
    # Capture frames from webcam for 3 seconds with lower resolution
    frames = capture_webcam_frames(duration_seconds=3, fps=4, width=320, height=240)
    
    if not frames:
        print("No frames captured. Exiting.")
        return
    
    # Save frames as temporary images
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = f"webcam_frame_{i}.jpg"
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(Path(frame_path))
        print(f"Saved frame {i+1} as {frame_path}")
    
    # Load model and processor
    model_id = "ezelanza/llava-next-video-openvino-int8"
    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
    model = OVModelForVisualCausalLM.from_pretrained(model_id)
    
    # Use frames as images in conversation
    conversation_with_frames = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what you see in these images. What is happening?"},
                *[{"type": "image", "image": path.as_posix()} for path in frame_paths],
            ],
        },
    ]
    
    # Process with the same model and processor
    inputs_with_frames = processor.apply_chat_template(
        conversation_with_frames,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True
    )
    
    # Generate response
    out_with_frames = model.generate(**inputs_with_frames, max_new_tokens=60)
    
    response_with_frames = processor.batch_decode(
        out_with_frames,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    
    print(f"MODEL OUTPUT (webcam frames): {response_with_frames}")
    
    if "ASSISTANT:" in response_with_frames:
        description_with_frames = response_with_frames.split("ASSISTANT:")[-1].strip()
    else:
        description_with_frames = response_with_frames.strip()
    
    print(f"CAPTION GENERATED (webcam frames): {description_with_frames}")
    
    # Clean up temporary frame files
    for frame_path in frame_paths:
        if frame_path.exists():
            frame_path.unlink()
            print(f"Cleaned up: {frame_path}")

if __name__ == "__main__":
    main() 