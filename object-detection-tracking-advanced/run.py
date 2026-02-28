import argparse
from pathlib import Path

from core.settings_loader import load_settings
from core.logging_setup import setup_logging
from src.pipeline import DetectionPipeline


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Object Detection and Tracking Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="inputs/videos/videoplayback.mp4",
        help="Path to input video or image"
    )
    parser.add_argument(
        "--counter",
        type=str,
        default="lane",
        choices=["line", "zone", "lane"],
        help="Type of counter to use"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N frames (default: 1)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display output window"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="Filter specific classes (e.g., --classes 0 2 for person and car)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_settings()
    setup_logging()
    
    # Override classes if specified
    if args.classes:
        config['detector']['classes'] = args.classes
    
    # Initialize pipeline
    pipeline = DetectionPipeline(config)
    
    # Determine input type
    input_path = Path(args.input)
    
    # Configure counter parameters (customize as needed)
    counter_kwargs = {}
    
    if args.counter == "line":
        counter_kwargs = {
            "start": (100, 200),
            "end": (500, 200)
        }
    elif args.counter == "zone":
        counter_kwargs = {
            "top_left": (100, 100),
            "bottom_right": (500, 300)
        }
    elif args.counter == "lane":
        counter_kwargs = {
            "points": [(180, 100), (400, 100), (550, 300), (50, 300)]
        }
    
    # Run pipeline
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        pipeline.run_video(
            video_path=input_path,
            counter_type=args.counter,
            counter_kwargs=counter_kwargs,
            vid_stride=args.stride,
            display=not args.no_display
        )
    elif input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        pipeline.run_image(
            image_path=input_path,
            counter_type=args.counter,
            counter_kwargs=counter_kwargs,
            display=not args.no_display
        )
    else:
        print(f"Unsupported file format: {input_path.suffix}")


if __name__ == "__main__":
    main()
