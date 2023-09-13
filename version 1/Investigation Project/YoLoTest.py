# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:08:28 2023

@author: 1892513
"""

from ultralytics import YOLO                        # YOLO model
from IPython.display import HTML, display, Image    # Image mananger and display
import subprocess

import os
HOME = os.getcwd()
print(HOME)

# Inference Results for Object Detection
# %cd {HOME}
# !yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='video.mp4' save=True

video_path = "video.mp4"

# Construct the YOLO command
yolo_command = f'yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source="{video_path}" save=True'

# Execute the YOLO command in PowerShell
subprocess.run(["powershell", "-Command", yolo_command], shell=True)