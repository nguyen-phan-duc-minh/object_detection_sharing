"""
Quick Start Example
Demonstrates basic usage of the detection pipeline
"""
from core.logging_setup import setup_logging, get_logger
from core.settings_loader import load_settings
from src.pipeline import DetectionPipeline

# Setup
setup_logging()
logger = get_logger("example")

# Load settings
settings = load_settings("configs/settings.yaml")

# Quick customization
settings.update('model.confidence_threshold', 0.4)
settings.update('visualization.show_display', True)

# Initialize pipeline
pipeline = DetectionPipeline(settings)

# Process video
logger.info("Processing video...")
summary = pipeline.process_video(
    video_path="inputs/videos/sample.mp4",
    output_path="outputs/videos/output.mp4"
)

# Display results
logger.info(f"Processed {summary['total_frames']} frames")
logger.info(f"Average FPS: {summary['average_fps']}")
logger.info(f"Total detections: {summary['total_detections']}")

if summary.get('line_counts'):
    logger.info("\nLine Counts:")
    for name, counts in summary['line_counts'].items():
        logger.info(f"  {name}: In={counts['in']}, Out={counts['out']}")

if summary.get('zone_counts'):
    logger.info("\nZone Counts:")
    for name, counts in summary['zone_counts'].items():
        logger.info(f"  {name}: In={counts['in']}, Out={counts['out']}, Current={counts['current']}")
