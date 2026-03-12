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
    
    input_path = Path(config["input"]["path"])
    vid_stride = config["input"]["vid_stride"]
    display = config["input"]["display"]
    
    pipeline = DetectionPipeline(config)
    counters = config.get("counters", [])
    
    if not counters:
        print("Warning: No counters configured in settings.yaml")
        return
    
    # Run pipeline for each counter
    for counter_config in counters:
        counter_type = counter_config["type"]
        counter_kwargs = {}
        
        if counter_type == "line":
            counter_kwargs = {
                "start": tuple(counter_config["start"]),
                "end": tuple(counter_config["end"])
            }
            if "color" in counter_config:
                counter_kwargs["color"] = counter_config["color"]
                
        elif counter_type == "zone":
            counter_kwargs = {
                "top_left": tuple(counter_config["top_left"]),
                "bottom_right": tuple(counter_config["bottom_right"])
            }
            if "color" in counter_config:
                counter_kwargs["color"] = counter_config["color"]
        elif counter_type in ["lane", "lanezone"]:
            points_data = counter_config["points"]
            
            # Kiểm tra xem có phải nhiều zones không (list 3 chiều)
            if len(points_data) > 0 and isinstance(points_data[0][0], (list, tuple)):
                # Nhiều zones: [[[x,y], [x,y]], [[x,y], [x,y]], ...]
                counter_kwargs = {
                    "points": [[tuple(point) for point in zone] for zone in points_data]
                }
            else:
                # Một zone duy nhất: [[x,y], [x,y], ...]
                counter_kwargs = {
                    "points": [tuple(point) for point in points_data]
                }
            
            if "colors" in counter_config:
                counter_kwargs["colors"] = counter_config["colors"]
            elif "color" in counter_config:
                counter_kwargs["colors"] = counter_config["color"]
            
            # Thêm max_speeds nếu có trong config
            if "max_speeds" in counter_config:
                counter_kwargs["max_speeds"] = counter_config["max_speeds"]
            
            counter_type = "lane"
        
        print(f"Running pipeline with {counter_type} counter...")
        
        # Run pipeline
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            pipeline.run_video(
                video_path=input_path,
                counter_type=counter_type,
                counter_kwargs=counter_kwargs,
                vid_stride=vid_stride,
                display=display
            )
        
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .jpg, .jpeg, .png, .bmp")
        
if __name__ == "__main__":
    main()