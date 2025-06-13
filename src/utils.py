from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, List, Union

def get_project_root() -> Path:
    
    return Path(__file__).parents[1]

def validate_video_path(path: str) -> str:
    
    full_path = get_project_root() / path
    if not full_path.exists():
        raise FileNotFoundError(f"Video file not found: {full_path}")
    return str(full_path)

def setup_output_directory(output_path: str) -> str:
    
    output_path = get_project_root() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)

def draw_bounding_box(frame: np.ndarray, 
                     bbox: Tuple[int, int, int, int], 
                     color: Tuple[int, int, int] = (0, 255, 0), 
                     thickness: int = 2) -> np.ndarray:
    
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame