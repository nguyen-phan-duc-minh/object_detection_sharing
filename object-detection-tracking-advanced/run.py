"""
Object Detection Application - Main Entry Point
Tracking + Counting System for Person and Car Detection
"""
import argparse
import sys
from pathlib import Path

from core.logging_setup import setup_logging, get_logger
from core.settings_loader import load_settings
from src.pipeline import DetectionPipeline


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Object Detection with Tracking and Counting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings
  python run.py --source inputs/videos/traffic.mp4
  
  # Process image
  python run.py --source inputs/images/test.jpg --output outputs/images/result.jpg
  
  # Process with custom config
  python run.py --source video.mp4 --config configs/custom.yaml
  
  # Process without display
  python run.py --source video.mp4 --no-display
  
  # Webcam (real-time)
  python run.py --source 0
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        help='Input source: video file, image file, or webcam (0)',
        required=False
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for processed video/image',
        required=False
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/settings.yaml',
        help='Path to configuration file (default: configs/settings.yaml)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable real-time display'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Disable saving output video/image'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        help='Confidence threshold (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help='Device to run inference on (overrides config)'
    )
    
    return parser.parse_args()


def validate_source(source: str) -> tuple[str, str]:
    """
    Validate and determine source type
    
    Args:
        source: Input source path or webcam index
        
    Returns:
        Tuple of (source_type, validated_source)
    """
    # Check if webcam
    if source.isdigit():
        return 'webcam', source
    
    # Check file existence
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    
    # Determine type from extension
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    ext = source_path.suffix.lower()
    
    if ext in video_exts:
        return 'video', str(source_path)
    elif ext in image_exts:
        return 'image', str(source_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def main():
    """Main application entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    logger = get_logger("main")
    
    logger.info("=" * 60)
    logger.info("Object Detection Pipeline - Starting")
    logger.info("=" * 60)
    
    try:
        # Load settings
        logger.info(f"Loading configuration from: {args.config}")
        settings = load_settings(args.config)
        
        # Override settings from command-line
        if args.no_display:
            settings.update('visualization.show_display', False)
        
        if args.no_save:
            settings.update('output.save_video', False)
            settings.update('output.save_images', False)
        
        if args.conf:
            settings.update('model.confidence_threshold', args.conf)
            logger.info(f"Confidence threshold: {args.conf}")
        
        if args.device:
            settings.update('model.device', args.device)
            logger.info(f"Device: {args.device}")
        
        # Determine source
        if args.source:
            source_type, source_path = validate_source(args.source)
            settings.update('input.source', source_path)
            settings.update('input.source_type', source_type)
        else:
            # Use source from config
            source_path = settings.get('input.source')
            source_type = settings.get('input.source_type', 'video')
            
            if not source_path:
                logger.error("No source specified. Use --source or configure in settings.yaml")
                sys.exit(1)
            
            if source_type != 'webcam':
                source_type, source_path = validate_source(source_path)
        
        logger.info(f"Source type: {source_type}")
        logger.info(f"Source: {source_path}")
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Auto-generate output path
            if source_type == 'video' or source_type == 'webcam':
                output_path = "outputs/videos/output.mp4"
            else:  # image
                output_path = "outputs/images/output.jpg"
        
        logger.info(f"Output: {output_path}")
        
        # Initialize pipeline
        pipeline = DetectionPipeline(settings)
        
        # Process based on source type
        logger.info("Starting processing...")
        
        if source_type == 'video' or source_type == 'webcam':
            # Handle webcam
            if source_type == 'webcam':
                source_path = int(source_path)
            
            summary = pipeline.process_video(source_path, output_path)
        else:  # image
            summary = pipeline.process_image(source_path, output_path)
        
        # Display summary
        logger.info("=" * 60)
        logger.info("Processing Summary")
        logger.info("=" * 60)
        
        if 'total_frames' in summary:
            logger.info(f"Total Frames: {summary['total_frames']}")
            logger.info(f"Average FPS: {summary['average_fps']}")
        
        logger.info(f"Total Detections: {summary.get('total_detections', 'N/A')}")
        
        if summary.get('line_counts'):
            logger.info("\nLine Counts:")
            for name, counts in summary['line_counts'].items():
                logger.info(f"  {name}: In={counts['in']}, Out={counts['out']}, Total={counts['total']}")
        
        if summary.get('zone_counts'):
            logger.info("\nZone Counts:")
            for name, counts in summary['zone_counts'].items():
                logger.info(f"  {name}: In={counts['in']}, Out={counts['out']}, Current={counts['current']}")
        
        logger.info("=" * 60)
        logger.info("Processing completed successfully!")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
