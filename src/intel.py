import pyrealsense2 as rs
import json

def init_cam_color(res=(848,480), fps=60, json_cfg='config1.json'):
    json_config = load_config_from_json(json_cfg)
    # Define pipeline for each camera
    pipeline_d435i = rs.pipeline()
    pipeline_d435 = rs.pipeline()

    # Configure both pipelines to connect to specific cameras
    config_d435i = rs.config()
    config_d435 = rs.config()

    serial_number_d435i, serial_number_d435 = ['152222070695', '153122073043']

    # Add serial numbers for each camera
    config_d435i.enable_device(serial_number_d435i)
    config_d435.enable_device(serial_number_d435)

    # Enable color streams
    config_d435i.enable_stream(rs.stream.color, res[0], res[1], rs.format.bgr8, fps)
    config_d435.enable_stream(rs.stream.color, res[0], res[1], rs.format.bgr8, fps)

    # Start both pipelines
    pipeline_d435.start(config_d435)
    pipeline_d435i.start(config_d435i)

    device_d435 = pipeline_d435.get_active_profile().get_device()
    device_d435i = pipeline_d435i.get_active_profile().get_device()

    advanced_mode_d435 = rs.rs400_advanced_mode(device_d435)
    advanced_mode_d435i = rs.rs400_advanced_mode(device_d435i)

    advanced_mode_d435i.load_json(json_config)
    advanced_mode_d435.load_json(json_config)

    return pipeline_d435i, pipeline_d435

def load_config_from_json(json_filepath):
    # Read the JSON file
    with open(json_filepath, 'r') as json_file:
        config_json = json.load(json_file)
    
    return json.dumps(config_json)

if __name__ == '__main__':
    x = (234234,234234)
    print(x[0])