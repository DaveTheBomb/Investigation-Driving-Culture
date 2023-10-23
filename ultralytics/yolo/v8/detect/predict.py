import hydra
import torch
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from predictor import BasePredictor
import os
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None


object_counter = {}

object_counter1 = {}

#Horizontal Line
line = [(100, 500), (1050, 500)] # test 1 video
#line = [(0, 500), (1280, 500)] # Stock video

# Tracking speeds
speed_line_queue = {}

def HorizontalLine_Before_Detection(img,line,color):
    cv2.line(img, line[0], line[1], color, 3)# Test1 video
    cv2.line(img, line[0], line[1], color, 3)# Stock  video
def HorizontalLine_After_Detection(img,line,color):
    cv2.line(img, line[0], line[1],color, 3)  # Test1 video
    cv2.line(img, line[0], line[1],color, 3)  # Stock video


def draw_transparent_yellow_rhombus(img):
    # Define the color of the filled rhombus in BGR-A format (yellow with transparency)
    alpha=100
    #vertices = [(550, 440), (635, 400), (1050, 430), (1010, 480]
    #Footage 1
    #vertices = [(300, 600), (430, 400), (860, 400), (1050, 600)]
    
    #Footage 2
    #vertices= [(120, 200), (75, 150), (560, 120), (700, 150)]

    #Footage 3
    #vertices= [(550, 440), (635, 400), (1050, 430), (1010, 480)]

    #Footage 4 
    vertices = [(750, 570), (730, 400), (780, 400), (850, 570)]

    yellow_color = (0, 255, 255, alpha)  # Yellow color with transparency (B, G, R, A)

    # Convert the vertices to a NumPy array
    vertices_np = np.array([vertices], dtype=np.int32)

    # Create an image with a transparent yellow filled rhombus
    filled_rhombus = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(filled_rhombus, vertices_np, yellow_color)

    # Blend the filled rhombus with the frame to add transparency
    img = cv2.addWeighted(filled_rhombus, 1, img, 1, 0)
    #Draw lines 
    lines = [((500, 440),(1000, 480)), ((600, 390),(1100, 430))] 
    line_color = [0, 0, 0]
    for line in lines:
      start_point, end_point = line
      line_thickness = 2
      cv2.line(img, start_point, end_point, line_color, line_thickness)
    
    return 
"""
def estimatespeed(Location1, Location2):
    #Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining thr pixels per meter
    ppm = 8
    d_meters = d_pixel/ppm
    time_constant = 15*3.6
    #distance = speed/time
    speed = d_meters * time_constant
    return int(speed)
"""


# Set the distance between the two virtual lines in meters
#STARTING LINES 

#Roadline = [(300, 600), (430, 400), (860, 400), (1050, 600)] #Footage 1
#Roadline = [(120, 200), (75, 150), (560, 120), (700, 150)] #Footage 2
#Roadline = [(550, 440), (635, 400), (1050, 430), (1010, 480)] # Footage 3
#Roadline = [(750, 570), (730, 400), (780, 400), (850, 570)] # Footage 4

#GRID-BASED METHODS 
#Footage 1 

Roadline = [
    ((300, 600), (1050, 600)),
    ((350, 550), (1000, 550)),
     ((380, 500), (950, 500)),
     ((410, 450), (920, 450)),
     ((440, 400), (890, 400)),
     ((470, 350), (860, 350)),
     ((475, 300), (830, 300)),
     ((485, 250), (800, 250)),
     ((510, 200), (770, 200)),]

#Footage 2
"""
Roadline = [
    ((-15, 60), (360,60)),
    ((30, 100), (450, 90)),
    ((75, 150), (560,120)),
    ((120, 200), (750, 150)),
    
]
"""
#Footage 3
"""
Roadline  = [
    ((500, 440), (1010, 480)),
    ((525, 415), (1035, 455)),
    ((550, 390), (1060, 430)),
    ((575, 365), (1085, 405)),
    ((600, 340), (1110, 380)),
    ((625, 315), (1135, 355)),
    ((650, 290), (1160, 330)),
    ((675, 265), (1185, 305)),
    ((700, 240), (1210, 280)),
    ((725, 215), (1235, 255))
]
"""

#Footage 4
"""
lines_to_draw = [
    ((660, 570), (660,400)),
    ((750, 570), (730,400)),
    ((780, 400), (850, 570)),
    ((830, 400), (950, 570))
 """  
def calculate_distance(center, point1, point2):
    # Calculate the distance between the vehicle's center and the line segment defined by point1 and point2
    # Use your distance calculation method here, for example, Euclidean distance
    # Calculate distance from the center to the line segment
    x1, y1 = point1
    x2, y2 = point2
    x, y = center
    distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return distance

def get_region_and_distance(center, roadlines):
    # Define the regions using the provided roadlines
    regions = ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5', 'Region 6', 'Region 7', 'Region 8', 'Region 9']
    distances = [None] * len(regions)

    for i in range(len(roadlines)):
        line = roadlines[i]
        if intersect(center, line[0], line[1]):
            # Vehicle is in the current region
            distances[i] = calculate_distance(center, line[0], line[1])

    return regions, distances

"""

# Function to calculate speed
def estimatespeed(direction, dist):
    if direction == "South" : 
        # Define the first two points in Roadline
        point1 = Roadline[0]
        point2 = Roadline[1]

        
        # Calculate the distance using the Euclidean distance formula
        #distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    else:
        point1 = Roadline[2]
        point2 = Roadline[3]

        

        # Calculate the distance using the Euclidean distance formula
       # distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        
    distance  = dist
    try:
        with open("time.txt", "r") as file:
            start_time = float(file.read().strip())
    except FileNotFoundError:
        print("Error: 'time.txt' file not found.")
        start_time = time.time()

    # Simulate vehicle crossing the start and end lines
    end_time = time.time()

    # Calculate the time taken in seconds
    time_taken = end_time - start_time
    # Calculate the speed in meters per second (m/s)
    speed = distance / time_taken
    # Convert speed to kilometers per hour (km/h)
    speed_kmph = speed * 3.6
    return speed_kmph

"""
import math
def estimatespeed(Location1, Location2, img, starting_time, ending_time):
     
     # Calculate the Euclidean distance between Location1 and Location2
    #d_pixel = math.sqrt((Location2[0] - Location1[0]) ** 2 + (Location2[1] - Location1[1]) ** 2)

    # Calculate the direction vector between Location1 and Location2 (Get the heright of the box)
    direction_vector = np.array([Location2[0] - Location1[0], Location2[1] - Location1[1]])
    # Calculate a perpendicular vector by swapping and negating the components 
    perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])

     # Calculate the magnitude of the perpendicular vector (To get width of the box)
    perpendicular_magnitude = np.linalg.norm(perpendicular_vector)

    # Check for zero magnitude to avoid division by zero
    if perpendicular_magnitude < 1e-6:
         return 0 
    
    # Calculate the starting and ending points for the perpendicular line
    line_start = (int(Location1[0] + perpendicular_vector[0]), int(Location1[1] + perpendicular_vector[1]))
    line_end = (int(Location2[0] + perpendicular_vector[0]), int(Location2[1] + perpendicular_vector[1]))
    
    # Use location1, Location2, line_start, line_end as a rectangle and apply perpendicular transformation
    pts1 = np.float32([Location1, Location2, line_end, line_start])
    pts2 = np.float32([[0, 0], [1, 0], [0, 600], [1, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (200, 600))

    if result.shape[1] >= result.shape[2]: #Check if width of transformed image is greater than the height
        d_pixel = result.shape[1] 
    else:
        d_pixel = result.shape[2] 


    ppm = 38 # Footage 1
    #ppm = 42 # Footage 2 
    #ppm = 22 # Footage 3 
    #ppm = 49 # Footage 4 
  
    # Convert the distance from pixels to meters
    d_meters = d_pixel / ppm
    
    # Calculate the time difference in seconds
    with open(os.path.join("Process_Delay.txt"), "r") as delay_file:
        # Read the processing delay from the file (assuming it contains a single float value per line)
        processing_delay_ms = float(delay_file.readline().strip())

    # Calculate the time difference in seconds
    time_difference = ending_time - starting_time - (processing_delay_ms / 1000)  # Subtract the process delay in seconds (converted from milliseconds)

    if time_difference == 0:
        return 0

    # Calculate the speed in meters per second
    speed = d_meters / time_difference
    #Convert m/s to Km/hr
    speed = speed*3.6
    
    return int(speed)

def determine_headlight_status(img, x1, y1, x2, y2, direction):
    # Crop the region of interest (ROI) from the image using (x1, y1, x2, y2)
    roi = img[y1:y2, x1:x2]

    # Define color ranges for red, orange, and green
    red_lower = np.array([0, 0, 100])
    red_upper = np.array([0, 0, 255])
    orange_lower = np.array([0, 100, 200])
    orange_upper = np.array([80, 180, 255])

    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Threshold the ROI to get color regions
    red_mask = cv2.inRange(hsv_roi, red_lower, red_upper)
    orange_mask = cv2.inRange(hsv_roi, orange_lower, orange_upper)
    # Calculate the percentage of red pixels in the ROI
    total_pixels = roi.shape[0] * roi.shape[1]
    red_pixel_percentage = (np.count_nonzero(red_mask) / total_pixels) * 100
    orange_pixel_percentage = (np.count_nonzero(orange_mask) / total_pixels) * 100

    if direction == "Northwest": #For footage 2. For footage 1 and 3, modify to north 
        if red_pixel_percentage >= 5.0 or orange_pixel_percentage >= 5.0:  
            headlight_status = "Taillights On"
        else:
            headlight_status = "Taillights Off"
    else:
        if red_pixel_percentage >= 1.0 or orange_pixel_percentage >= 1.0:  
            headlight_status = "Headlights On"
        else:
            headlight_status = "Headlights Off"
    
    color = (255, 255, 255)
    text_x = int((x1 + x2) / 2)
    text_y = y2 + 20
    cv2.putText(img, headlight_status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, lineType=cv2.LINE_AA)

#Detecting the traffic light 
def detect_traffic_light_color(image, x1, y1, x2, y2):
    # Crop the region of interest (ROI) from the image using (x1, y1, x2, y2)
    roi = image[y1:y2, x1:x2]

    # Define color ranges for red, orange, and green
    red_lower = np.array([0, 0, 100])
    red_upper = np.array([100, 100, 255])
    
    orange_lower = np.array([0, 100, 200])
    orange_upper = np.array([80, 180, 255])
    
    green_lower = np.array([0, 100, 0])
    green_upper = np.array([80, 255, 80])
    
    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Threshold the ROI to get color regions
    red_mask = cv2.inRange(hsv_roi, red_lower, red_upper)
    orange_mask = cv2.inRange(hsv_roi, orange_lower, orange_upper)
    green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)
    
    # Count the number of non-zero pixels in each mask
    red_pixel_count = np.count_nonzero(red_mask)
    orange_pixel_count = np.count_nonzero(orange_mask)
    green_pixel_count = np.count_nonzero(green_mask)
    
    # Determine the detected color based on the pixel counts
    if red_pixel_count > orange_pixel_count and red_pixel_count > green_pixel_count:
        return "Red"
    elif orange_pixel_count > red_pixel_count and orange_pixel_count > green_pixel_count:
        return "Orange"
    elif green_pixel_count > red_pixel_count and green_pixel_count > orange_pixel_count:
        return "Green"
    else:
        return "Unknown"


def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

#DRAW  RECTANGLE AROUND DETECTED OBJECT AND ANNOTATE WITH ID AND OBJECT NAME
def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str

#DRAW TRAILS LINES 
def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    height, width, _ = img.shape

    color = (255, 165, 0) # Dark Green 
    HorizontalLine_Before_Detection(img,line,color)    

    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))
      

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
          speed_line_queue[id] = []
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = "#"+'{}{:d}'.format("", id) + " "+ '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
############################################################################
        # Record the initial time and write it to the time.txt file
        line_1  = (Roadline[0], Roadline[1])
        line_2  = (Roadline[0], Roadline[1])

        # Record the initial time and write it to the time.txt file
        if intersect(center, line_1[0], line_1[1]) or intersect(center, line_2[0], line_2[1]):
            initial_time = time.time()
            with open("time.txt", "w") as file:
                file.write(str(initial_time))
        
##############################################################################
         # Display the traffic light color next to the traffic light
        if obj_name == "traffic light":
            traffic_light_color = detect_traffic_light_color(img, x1, y1, x2, y2)
            if traffic_light_color:
                cv2.putText(img, f'Light: {traffic_light_color}', (x1, y2 +15), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 255, 255), 1, lineType=cv2.LINE_AA)

        

        if len(data_deque[id]) >= 2:
          direction = get_direction(data_deque[id][0], data_deque[id][1])
          #object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
          # Call the estimatespeed function within draw_boxes
          #_, distance = get_region_and_distance(center, Roadline)
          #object_speed = estimatespeed(direction, distance)
          object_speed = estimatespeed(data_deque[id][1][0], data_deque[id][0][0], data_deque[id][1][1], data_deque[id][0][1])
          if obj_name != "traffic light":
             determine_headlight_status(img, x1, y1, x2, y2, direction)
          
          speed_line_queue[id].append(object_speed)
          if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
              color = (255, 255, 255)
              HorizontalLine_After_Detection(img,line,color)
              if "South" in direction:
                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1
        
        try:
            label = label + " " + str(sum(speed_line_queue[id])//len(speed_line_queue[id])) + "km/hr"
        except:
            pass
        
        UI_box(box, img, label=label, color=color, line_thickness=1)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
         
         #Display Count in top right corner
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = str(key) + ":" +str(value)
            cv2.line(img, (width - 500,25), (width,25), [85,45,200], 40)
            cv2.putText(img, f'Number of Vehicles Entering; {sum(object_counter1.values())}', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img, (width - 150, 65 + (idx*40)), (width, 65 + (idx*40)), [85,45,200], 30)
            cv2.putText(img, cnt_str, (width - 150, 75 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)
        
        # Display Count in top left  corner
        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str1 = str(key) + ":" +str(value)
            cv2.line(img, (20,25), (500,25), [85,45,200], 40)
            cv2.putText(img, f'Numbers of Vehicles Leaving: {sum(object_counter.values())}', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
            cv2.line(img, (20,65+ (idx*40)), (127,65+ (idx*40)), [85,45,200], 30)
            cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            
    return img


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        #IMPLEMENTING THE DEEPSORT ALGORITHM  
        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
