import cv2
from huggingface_hub import hf_hub_download

# Download video
video_path = hf_hub_download(
    repo_id="raushan-testing-hf/videos-test", 
    filename="sample_demo_1.mp4", 
    repo_type="dataset"
)

# Open video and get properties
cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video file: {video_path}")
    print(f"Original resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.1f} seconds")
    
    cap.release()
else:
    print("Could not open video file") 