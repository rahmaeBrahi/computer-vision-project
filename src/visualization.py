import cv2
import os
from tracker import SinglePersonTracker

def process_video(input_path: str, output_path: str,model:str) -> None:
   
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    
    tracker = SinglePersonTracker(model_path=model)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    codecs = ['mp4v', 'XVID', 'MJPG', 'avc1']
    out = None
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec}")
                break
        except:
            continue
    
    if out is None:
        raise RuntimeError("Failed to create video writer")
   
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            frame = tracker.track_single_person(frame)
            out.write(frame)
            frame_count += 1
            print(f"Processed frame {frame_count}", end='\r')
        except Exception as e:
            print(f"\nFrame {frame_count} error: {str(e)}")
            continue

    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nFinished! Processed {frame_count} frames")
    print(f"Output saved to: {output_path}")