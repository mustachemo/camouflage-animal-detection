import cv2

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder to save extracted frames.
        frame_rate (int): Extract 1 frame per second (default: 1).

    Returns:
        List of file paths of the extracted frames.
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frame_files = []
    
    while success:
        if int(vidcap.get(cv2.CAP_PROP_POS_MSEC)) % 1000 < frame_rate * 1000:
            frame_file = f"{output_folder}/frame_{count}.jpg"
            cv2.imwrite(frame_file, image)
            frame_files.append(frame_file)
            count += 1
        success, image = vidcap.read()
    
    return frame_files
