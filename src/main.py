import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
from visualization import process_video

def main():
    

    models = ["yolov5s"]#,"yolo11m","yolo11x"]
    for m in models:
        for video in ["input", "input2","video2"]:
            
            print(f"Starting processing {video} using model: {m}")
            process_video(f"data/raw/{video}.mp4",f'data/processed/{video}_{m}.mp4' , model=f'../models/{m}.pt')
            print(f"Finished processing video using model: {m}")
if __name__ == '__main__':
    main()