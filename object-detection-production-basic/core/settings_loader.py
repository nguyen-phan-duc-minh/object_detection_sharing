import yaml
from pathlib import Path
from dotenv import load_dotenv
import os

def load_settings():
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        
    config_path = Path("config/settings.yaml")
    with open(config_path) as file:
        settings = yaml.safe_load(file)
        
   # App settings
    if os.getenv("APP_ENV"):
        settings["app"]["env"] = os.getenv("APP_ENV")
    if os.getenv("APP_NAME"):
        settings["app"]["name"] = os.getenv("APP_NAME")
    
    # Data paths
    if os.getenv("DATA_SOURCE"):
        settings["data"]["source"] = os.getenv("DATA_SOURCE")
    if os.getenv("DATA_OUTPUT"):
        settings["data"]["output"] = os.getenv("DATA_OUTPUT")
    if os.getenv("DATA_MODEL"):
        settings["data"]["model"] = os.getenv("DATA_MODEL")
    
    # Detector settings
    if os.getenv("DETECTOR_TYPE"):
        settings["detector"]["type"] = os.getenv("DETECTOR_TYPE")
    if os.getenv("DETECTOR_CONF_THRESHOLD"):
        settings["detector"]["conf_threshold"] = float(os.getenv("DETECTOR_CONF_THRESHOLD"))
    if os.getenv("DETECTOR_IOU_THRESHOLD"):
        settings["detector"]["iou_threshold"] = float(os.getenv("DETECTOR_IOU_THRESHOLD"))
    if os.getenv("DETECTOR_MAX_DET"):
        detector_max_det_str = os.getenv("DETECTOR_MAX_DET")
        if detector_max_det_str.lower() != "null":
            settings["detector"]["max_det"] = int(detector_max_det_str)
    if os.getenv("DETECTOR_DEVICE"):
        device_str = os.getenv("DETECTOR_DEVICE")
        settings["detector"]["device"] = None if device_str.lower() == "null" else device_str
    if os.getenv("DETECTOR_CLASSES"):
        # Parse comma-separated class IDs
        classes_str = os.getenv("DETECTOR_CLASSES")
        if classes_str.lower() != "null":
            settings["detector"]["classes"] = [int(x.strip()) for x in classes_str.split(",")]
        else:
            settings["detector"]["classes"] = None
    
    # Input settings
    if os.getenv("INPUT_PATH"):
        settings["input"]["path"] = os.getenv("INPUT_PATH")
    if os.getenv("INPUT_VID_STRIDE"):
        settings["input"]["vid_stride"] = int(os.getenv("INPUT_VID_STRIDE"))
    if os.getenv("INPUT_DISPLAY"):
        settings["input"]["display"] = os.getenv("INPUT_DISPLAY").lower() == "true"
    
    # Saver settings
    if os.getenv("SAVER_SAVE_IMAGES"):
        settings["saver"]["save_images"] = os.getenv("SAVER_SAVE_IMAGES").lower() == "true"
    if os.getenv("SAVER_SAVE_DETECTIONS"):
        settings["saver"]["save_detections"] = os.getenv("SAVER_SAVE_DETECTIONS").lower() == "true"
    if os.getenv("SAVER_SAVE_CROPS"):
        settings["saver"]["save_crops"] = os.getenv("SAVER_SAVE_CROPS").lower() == "true"
    
    # Tracker settings
    if os.getenv("TRACKER_TYPE"):
        settings["tracker"]["type"] = os.getenv("TRACKER_TYPE")
    
    return settings