import cv2
import os 

def extract_frames(video_path, output_folder):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder to save extracted frames.
        

    Returns:
        Images of the extracted frames.
    """
    frame_interval = 10  # Save every 10th frame
    max_duration = 10    # Limit to first 10 seconds of video
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(int(fps * max_duration), total_frames)
    
    
    # Process frames
    while frame_count < max_frames:
        success, frame = video_capture.read()
        if not success:
            break  # Exit if no more frames
        
        # Save every 10th frame
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1
        
        frame_count += 1

    # Release video capture
    video_capture.release()
    print(f"Total frames saved: {saved_frame_count}")

# # Example usage
# video_path = "istockphoto-1471847063-640_adpp_is.mp4"  # Path to the input video
# output_folder = "frames_output"  # Folder to save frames
# extract_frames(video_path, output_folder)