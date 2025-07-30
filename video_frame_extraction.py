# Extract frames from video file for LLaVA model
# This demonstrates how to extract frames from a video and use them as images

import cv2
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

def extract_video_frames(video_path, num_frames=4, width=320, height=240):
    """Extract evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return []
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_video_frames / fps
    
    print(f"Video info: {total_video_frames} frames, {fps:.1f} FPS, {duration:.1f} seconds")
    
    # Calculate frame indices to extract (evenly spaced)
    frame_indices = np.linspace(0, total_video_frames-1, num_frames, dtype=int)
    
    frames = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Resize frame to reduce processing time
            frame_resized = cv2.resize(frame, (width, height))
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            print(f"Extracted frame {i+1}/{num_frames} at frame {frame_idx}")
    
    cap.release()
    return frames

def main():
    # Download video
    video_path = hf_hub_download(
        repo_id="raushan-testing-hf/videos-test", 
        filename="sample_demo_1.mp4", 
        repo_type="dataset"
    )
    
    print(f"Downloaded video: {video_path}")
    
    # Extract frames from video
    frames = extract_video_frames(video_path, num_frames=4, width=320, height=240)
    
    if not frames:
        print("No frames extracted. Exiting.")
        return
    
    # Save frames as temporary images
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = f"video_frame_{i}.jpg"
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
    
    print(f"MODEL OUTPUT (video frames): {response_with_frames}")
    
    if "ASSISTANT:" in response_with_frames:
        description_with_frames = response_with_frames.split("ASSISTANT:")[-1].strip()
    else:
        description_with_frames = response_with_frames.strip()
    
    print(f"CAPTION GENERATED (video frames): {description_with_frames}")
    
    # Clean up temporary frame files
    for frame_path in frame_paths:
        if frame_path.exists():
            frame_path.unlink()
            print(f"Cleaned up: {frame_path}")

if __name__ == "__main__":
    main() 