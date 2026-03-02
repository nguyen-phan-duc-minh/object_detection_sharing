from pathlib import Path

from core.settings_loader import load_settings
from core.logging_setup import setup_logging
from src.pipeline import DetectionPipeline

def main():
    config = load_settings()
    setup_logging()
    
    Path("inputs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    INPUT_PATH = "inputs/videos/videoplayback.mp4"
    COUNTER_TYPE = "lane"  
    VID_STRIDE = 1  
    DISPLAY = True 
    
    pipeline = DetectionPipeline(config)
    input_path = Path(INPUT_PATH)
    counter_kwargs = {}
    
    if COUNTER_TYPE == "line":
        counter_kwargs = {"start": (100, 200), "end": (500, 200)}
    elif COUNTER_TYPE == "zone":
        counter_kwargs = {"top_left": (100, 100), "bottom_right": (500, 300)}
    elif COUNTER_TYPE == "lane":
        counter_kwargs = {"points": [(180, 100), (400, 100), (550, 300), (50, 300)]}
    
    # Run pipeline
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        pipeline.run_video(
            video_path=input_path,
            counter_type=COUNTER_TYPE,
            counter_kwargs=counter_kwargs,
            vid_stride=VID_STRIDE,
            display=DISPLAY
        )
        
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .jpg, .jpeg, .png, .bmp")
        
if __name__ == "__main__":
    main()