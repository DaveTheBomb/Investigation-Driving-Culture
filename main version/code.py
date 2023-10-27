import cv2
import numpy as np
from ultralytics import YOLO
import random
import pygame
# import threading
# import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from csv import writer, DictWriter


class Vehicle:
    def __init__(self):
        self.average_following_distance = 0
        self.current_following_distance = 0
        self.vehicle_class = None
        self.id = -1
        self.number_of_line_changes = -1
        self.line_change_history = []
        self.initial_lane = -1
        self.initial_lane_index = -1 
        self.time_stampes = []
        self.position =  (- 1, -1)
        self.following_distance_sum = 0
        self.counts = 0
        self.following_vehicle_with_id = -1

    def print(self):
        print("..............................................................")        
        print("ID:                         ", self.id)
        print("Type:                       ", self.vehicle_class)
        print("Average Following Distance: ", self.average_following_distance)
        print("Line History:               ", self.line_change_history)
        print("..............................................................")

class DataCollection:
    def __init__(self):
        self.vehicle_data = {}

class ObjectDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        results = self.model(frame)
        return results



class PerspectiveTransformation:
    def __init__(self, orignal_first_frame, width_, height_, perspective_points):
        self.width = width_
        self.height = height_
        self.original_first_frame = orignal_first_frame # unscaled image

        # pointsy = [((319, 241), (455, 237)), ((848, 584), (113, 581))]


        # *** DO NOT DELETE ***  MOVE TO OUT SCOPE
        # points_drawer = Process(self.original_first_frame, self.width, self.height)
        # pointsy = points_drawer.getLines()
        
        pointsy = perspective_points


        print("Transformation points:", pointsy)

        line1 = pointsy[0]
        line2 = pointsy[1]
        coordinates = [list(line1[0]), list(line1[1]), list(line2[0]), list(line2[1])]
        self.matrix = self.generateTranformationMatrix(coordinates)

        return
        
    def getTransformedImage(self, image_path, matrix):
        image = cv2.imread(image_path)
        radius = 10
        
        # visualizing the points
        # cv2.circle(image, (x1, y1), radius, (255, 0, 0), -1)
        # cv2.circle(image, (x2, y2), radius, (255, 0, 0), -1)

        # Apply the perspective transformation
        result = cv2.warpPerspective(image, matrix, (self.width, self.height))

        # Display the original and transformed images
        cv2.imshow("Original Image", image)
        cv2.imshow("Bird's Eye View", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return



    def generateTranformationMatrix(self, coordinates):
        width = self.width
        height = self.height
        pts_src = np.array(coordinates, dtype=np.float32)
        pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        return matrix

    def caculateActualPosition(self, matrix, point_1, point_2):
        point1 = np.array([point_1[0], point_1[1], 1])
        point2 = np.array([point_2[0], point_2[1], 1])

        transformed_point1 = np.dot(matrix, point1)
        transformed_point2 = np.dot(matrix, point2)

        transformed_x1 = transformed_point1[0]/ transformed_point1[2]
        transformed_y1 = transformed_point1[1]/ transformed_point1[2]
        transformed_x2 = transformed_point2[0]/ transformed_point2[2]
        transformed_y2 = transformed_point2[1]/ transformed_point2[2]

        new_cordinates = [(transformed_x1, transformed_y1), (transformed_x2, transformed_y2)]
        point1_t = np.array([transformed_x1, transformed_y1])
        point2_t = np.array([transformed_x2, transformed_y2])

        ratio = 300/12.712
        pixel_distance = np.linalg.norm(point1_t - point2_t)
        eucludien_distance = pixel_distance/ratio
        return eucludien_distance


class ObjectTracker:
    def __init__(self, img_path, width_, height_, perspective_points):    
        self.tracker = None
        self.tracker_initialized = False
        self.tracker_boxes = None
        self.tracker_ids =  None
        self.tracker_names = None
        self.lanes_tracker = {}
        self.lane_tracker_created = False
        self.dictionary = {}
        self.BigData = DataCollection()
        self.Vehicles_collection = {}




        self.counumber = 0


        self.image_perspective_transformation = PerspectiveTransformation(img_path, width_, height_, perspective_points)
        self.matrix =  self.image_perspective_transformation.matrix

        # the following distance 
        self.following_distance_results = []

        self.headings_written = False
        self.my_results = []

    def initialize_tracker(self, model_path):
        self.tracker = YOLO(model_path)
        self.tracker_names = self.tracker.names
        self.tracker_initialized = True
        return
    
    def getAnnotations(self):
        #print("Annotations function: ", self.tracker_names)
        return [self.tracker_boxes, self.tracker_ids, self.tracker_classes]


    def point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            if (y1 == y2) and (y == y1) and (x >= min(x1, x2)) and (x <= max(x1, x2)):
                return True

            if (y1 < y and y2 >= y) or (y2 < y and y1 >= y):
                if x1 + (y - y1) / (y2 - y1) * (x2 - x1) < x:
                    inside = not inside

        return inside

    def customAnnotateFrame(self, frame, tracked_boxes, tracked_ids, tracked_classes, road_lanes):


        for i in range(len(road_lanes)):
            self.lanes_tracker[i+1] = []

        # print(self.lanes_tracker) 

        for tracked_box, track_id, tracked_class in zip(tracked_boxes, tracked_ids, tracked_classes):
            x, y, w, h = [int(i) for i in tracked_box]
            x_offset = int(w/2)
            y_offset = int(h/2)
            cv2.rectangle(frame, (x - x_offset , y - y_offset), (x + x_offset, y + y_offset), (0, 255, 0), 3)


            # DO NOT DELETE
            # if track_id == 1: 
                           
            #     self.counumber = self.counumber + 1
            #     cropped_image = frame[y - y_offset: y + y_offset, x - x_offset :x + x_offset, ]
            #     cv2.imwrite(f"allframes/cropped{self.counumber}.jpeg", cropped_image)



            initial_lane = None
            pointy = [x, int(y + y_offset)]
            
            for initial_lane, lane in enumerate(road_lanes):
                if self.point_in_polygon(pointy, lane) == True:
                    break
                initial_lane = -2

            changed_line = None
            following_vehicle_with_id = None
            following_distance = 0.000

            if (initial_lane + 1) > 0:
                if track_id not in self.lanes_tracker[initial_lane + 1]:
                    self.lanes_tracker[initial_lane + 1].append(track_id)

                    ith_vehicle = Vehicle()
                    ith_vehicle.position = (x, y - y_offset)
                    self.Vehicles_collection[track_id] = ith_vehicle
                    following_distance, following_vehicle_with_id = self.calculateFollowingDistance(initial_lane, track_id)

                    if track_id not in self.BigData.vehicle_data:
                        self.BigData.vehicle_data[track_id] = Vehicle()
                        self.BigData.vehicle_data[track_id].id = track_id


                    if track_id in self.BigData.vehicle_data:
                        if (initial_lane + 1) not in self.BigData.vehicle_data[track_id].line_change_history:
                            self.BigData.vehicle_data[track_id].line_change_history.append(initial_lane + 1)
                            self.BigData.vehicle_data[track_id].vehicle_class = self.tracker_names[int(tracked_class)]
                            self.BigData.vehicle_data[track_id].initial_lane_index = initial_lane + 1
                            
                        if (initial_lane + 1) in self.BigData.vehicle_data[track_id].line_change_history:
                            last_lane = self.BigData.vehicle_data[track_id].line_change_history[-1]
                            if last_lane != (initial_lane + 1):
                                self.BigData.vehicle_data[track_id].line_change_history.append(initial_lane + 1)

                        if following_distance > 0:
                            self.BigData.vehicle_data[track_id].counts = self.BigData.vehicle_data[track_id].counts + 1 
                            self.BigData.vehicle_data[track_id].following_distance_sum = self.BigData.vehicle_data[track_id].following_distance_sum + following_distance
                            self.BigData.vehicle_data[track_id].average_following_distance = self.BigData.vehicle_data[track_id].following_distance_sum/self.BigData.vehicle_data[track_id].counts
                            self.BigData.vehicle_data[track_id].following_vehicle_with_id = following_vehicle_with_id 


                    else:
                        # update
                        pass                        


                else:
                    print("Lane changed")
                    
            radius = 10
            cv2.circle(frame, (x, (y + y_offset)), radius, (255, 0, 0), -1)

            text_color = (255, 255, 255)
            font_scale = 0.5
            font_size = 2
            cv2.putText(frame, f"Vehicle class: {self.tracker_names[int(tracked_class)]}", (x - x_offset, y - y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_size)
            cv2.putText(frame, f"Following distance: {following_distance}", (x - x_offset, y - y_offset - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_size)
            cv2.putText(frame, f"Initial lane: {initial_lane + 1}", (x - x_offset, y - y_offset - 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_size)
            cv2.putText(frame, f"Vehicle followed #: {following_vehicle_with_id}", (x - x_offset, y - y_offset - 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_size)
            cv2.putText(frame, f"ID: {track_id}", (x - x_offset, y - y_offset - 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_size)


        # ** DO NOT DELETE **

        # if self.headings_written == False:
        #     headings = ['ID', 'Vehicle class', 'Following distance', 'Vehicle followed', 'Currency lane']
        #     with open('data/event.csv', mode='w', newline='') as f_object:
        #         writer_object = writer(f_object, delimiter=';')
        #         writer_object.writerow(headings)
        #         f_object.flush()
        #     self.headings_written = True



        # x_ith = []
        # y_ith = []
        file_data = []

        for key in self.BigData.vehicle_data:
            itm = self.BigData.vehicle_data[key]

            # print(itm.print())
            x_ith =itm.id
            y_ith = itm.average_following_distance
            y_ith = float('{0:.2f}'.format(y_ith))
            history_of_lane_change = [str(element) for element in itm.line_change_history] 
            history_of_lane_change = ','.join(str(item) for item in itm.line_change_history)
            v_class = itm.vehicle_class 


            field_names = ["ID", "VehicleClass" , "AverageFollowingDistance", "LaneChangeHistory"]

            # print(itm.id, " *** " , history_of_lane_change)

            file_data.append({'ID': x_ith, 'VehicleClass': v_class, 'AverageFollowingDistance': y_ith, 'LaneChangeHistory':history_of_lane_change})

            # print('Dictionary: ', file_data)

            with open('data/results.csv', mode ='w', newline = '') as f_results:
                writer = DictWriter(f_results, delimiter = ';', fieldnames = field_names)
                writer.writeheader()
                writer.writerows(file_data)
                f_results.flush()
            


        #     data = [itm.id, itm.vehicle_class, itm.average_following_distance, itm.following_vehicle_with_id, itm.initial_lane_index]
        #     with open('data/event.csv', mode='a', newline='') as f_object:
        #         writer_object = writer(f_object, delimiter=';')
        #         writer_object.writerow(data)
        #         f_object.flush()
            
        return frame

    def calculateFollowingDistance(self, initial_lane, track_id):
        following_distance = 0.00
        following_vehicle_with_id = None
        
        if (len(self.lanes_tracker[initial_lane + 1])) >= 2:
            following_vehicle_with_id = self.lanes_tracker[initial_lane + 1][-2]

            point_1 = self.Vehicles_collection[track_id].position
            point_2 = self.Vehicles_collection[following_vehicle_with_id].position

            distanc = self.image_perspective_transformation.caculateActualPosition(self.matrix, point_1, point_2)
            # print("Following distance: ", distanc)
            following_distance = distanc
            following_distance = float("{:.2f}".format(following_distance))
        
        return following_distance, following_vehicle_with_id


    def track_objects(self, frame): # to be fixed
        if self.tracker_initialized:
            results = self.tracker.track(frame, persist = True)
            if results[0].boxes is not None:
                self.tracker_classes = results[0].boxes.cls.tolist()
                self.tracker_boxes = results[0].boxes.xywh.cpu().tolist()
                # Check if results[0].boxes.id is not None before converting to int
                if results[0].boxes.id is not None:
                    self.tracker_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    self.tracker_ids = None






                annotated_frame = results[0].plot()
                return annotated_frame  
        else:
            print("Model not initialized")
            return frame 



class ImagePolygonDrawer:
    def __init__(self, image_frame):
        self.image = image_frame.copy()
        self.original_image = self.image.copy()

    def resize_image(self, width, height):
        self.image = cv2.resize(self.original_image, (width, height))

    def draw_polygon(self, points, color=(0, 0, 255), thickness=2, transparency=0.5):
        overlay = self.original_image.copy()  # Use the original image as a base for overlay
        cv2.fillPoly(overlay, [np.array(points)], color)
        cv2.addWeighted(overlay, transparency, self.image, 1 - transparency, 0, self.image)
        return self.image

class VideoFrameCapture:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

    def capture_first_frame(self):
        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read the first frame.")
            return None

        self.cap.release()
        return frame

class Process:
    def __init__(self, img_path, width, height):
        pygame.init()
        self.setup_pygame()
        self.width = width
        self.height = height
        self.background_image = self.load_background_image(img_path, width, height) 
        self.points = []
        self.lines = []
        self.run_game_loop()


    def setup_pygame(self):
        WIDTH, HEIGHT = 850, 700
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Draw Points and Lines")

    def load_background_image(self, image_path, width, height):
        background_image = pygame.image.load(image_path)
        return pygame.transform.scale(background_image, (self.width, self.height))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.points.append(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    return False
                if event.key == pygame.K_u:
                    self.lines = self.lines[:-1]
                    self.points = self.points[:-2]
        return True

    def draw_lines_and_points(self):
        POINT_COLOR = (0, 0, 0)
        POINT_RADIUS = 5
        LINE_COLOR = (255, 0, 0)
        LINE_WIDTH = 2

        self.screen.blit(self.background_image, (0, 0))

        for i in range(0, len(self.points) - 1, 2):
            pygame.draw.line(self.screen, LINE_COLOR, self.points[i], self.points[i + 1], LINE_WIDTH)
            line = (self.points[i], self.points[i + 1])
            if line not in self.lines:
                self.lines.append(line)

        for point in self.points:
            pygame.draw.circle(self.screen, POINT_COLOR, point, POINT_RADIUS)

        pygame.display.flip()

    def run_game_loop(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw_lines_and_points()

        pygame.quit()

    def getLines(self):
        return self.lines


class ImagePolygonDrawerFirstFrame:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.original_image = self.image.copy()

    def resize_image(self, width, height):
        self.image = cv2.resize(self.image, (width, height))

    def draw_polygon(self, points, color=(0, 0, 255), thickness=2, transparency=0.5):
        overlay = self.image.copy()
        cv2.fillPoly(overlay, [np.array(points)], color)
        cv2.addWeighted(overlay, transparency, self.image, 1 - transparency, 0, self.image)

    def save_image(self, output_path):
        cv2.imwrite(output_path, self.image)

    def display_image(self):
        cv2.imshow('Image with Polygon', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class Manager:
    def __init__(self):
        self.final_polygone_list = []

        self.output_path = None
        self.image_path = None
        self.video_path = None

        self.new_width = None
        self.new_height = None


    def setting(self, output_path_, image_path_, video_path_, screen_size):
        self.output_path = output_path_
        self.image_path = image_path_
        self.video_path = video_path_

        self.new_width = screen_size[0]
        self.new_height = screen_size[1]
        return



    def processVideo(self):
        video_capture = VideoFrameCapture(self.video_path)
        first_frame = video_capture.capture_first_frame()
        # save the path for easy access
        cv2.imwrite(self.image_path, first_frame) 
        background_image = first_frame

        if first_frame is not None:
            process = Process(self.image_path, self.new_width, self.new_height)

        image_drawer = ImagePolygonDrawerFirstFrame(self.image_path)
        image_drawer.resize_image(self.new_width, self.new_height)

        polygon_points = []
        lines = process.getLines()

        number_of_lines = 0
        current_color = 0


        color_palette = [(255, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (255, 255, 255), (0, 255, 255), (20, 255, 20), (230, 200, 255)]

        for line in lines:
            polygon_points.append(line[0])
            polygon_points.append(line[1])
            number_of_lines += 1

            if number_of_lines == 2:
                image_drawer.draw_polygon(
                    polygon_points, color=color_palette[current_color], thickness=2, transparency=0.5
                )

                formated_polygon_list = [list(point) for point in polygon_points]
                self.final_polygone_list.append(formated_polygon_list)

                polygon_points = []
                number_of_lines = 0
                current_color += 1



        # print(f'total number of lanes {len(self.final_polygone_list)}')

        #image_drawer.save_image(self.output_path)
        image_drawer.display_image()
        return

    def get_polygon_list(self):
        return self.final_polygone_list





class VideoProcessor:
    def __init__(self, model_path, output_path, output_frame_size, background_img, road_lanes = [], perspective_points = None):

        width__  = output_frame_size[0]
        height__  = output_frame_size[1]

        self.object_detector = ObjectDetection(model_path)
        self.object_tracker = ObjectTracker(background_img, width__, height__, perspective_points)
        self.object_tracker.initialize_tracker(model_path)
        self.road_lanes = road_lanes
        

        self.output_path = output_path
        self.output_frame_size = output_frame_size

    def process_frame(self, frame):
        # Perform object detection
        resized_frame = cv2.resize(frame, self.output_frame_size)
        #object_results = self.object_detector.detect_objects(resized_frame)
        #print("Object Detection Results", object_results)

        tracked_frame = self.object_tracker.track_objects(resized_frame)
        tracked_frame_annotations = self.object_tracker.getAnnotations()
        #print("Okay :", self.object_tracker.tracker_names)

        annotated_frame = resized_frame

        #print("Annoations: ***", tracked_frame_annotations)
        if None not in tracked_frame_annotations:
            tracked_frame = self.object_tracker.customAnnotateFrame(resized_frame, tracked_frame_annotations[0], tracked_frame_annotations[1],tracked_frame_annotations[2], self.road_lanes )
            annotated_frame = tracked_frame

        color_palete = [(255, 255, 0), (255, 0, 0), (255, 0, 255), (0 ,0 ,255)]
        color_palete = [(255, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (255, 255, 255), (0, 255, 255), (20, 255, 20), (230, 200, 255)]
        img =ImagePolygonDrawer(annotated_frame)  # Create a new ImagePolygonDrawer for each lane
        
        for index, lane in enumerate(self.road_lanes):
            annotated_frame = img.draw_polygon(lane, color_palete[index], 1, 0.3)
        
        return annotated_frame






class Video_settings:
    def __init__(self):
        
        self.footage_path =  None 
        self.polygones = None
        self.perspective_points = None
        self.video_path = None
        
        self.footage_choice = self.prompt_user()
        
    def prompt_user(self):
        footage_choice = input("Enter the footage to excute: \nList of footages: footage1, footage2, footage3. \nExample of input 'footage1', NOTE! do not include the quotations.\n")
        
        if footage_choice == "footage1":
            self.polygones = [[[157, 699], [362, 111], [374, 110], [344, 697]], [[358, 696], [375, 109], [384, 112], [539, 693]], [[555, 696], [385, 112], [399, 114], [729, 696]]]
            self.perspective_points = [((363, 111), (393, 111)), ((729, 696), (162, 697))]
            self.video_path = 'data/Footage1.mp4'
            return 
            
        if footage_choice == "footage2":
            self.polygones = [[[3, 47], [431, 662], [619, 573], [22, 45]], [[620, 569], [23, 45], [38, 42], [771, 486]], [[771, 486], [41, 41], [64, 40], [844, 412]], [[846, 412], [64, 42], [89, 39], [847, 346]], [[848, 345], [87, 39], [113, 36], [847, 292]]]
            self.perspective_points = [((3, 50), (95, 31)), ((846, 296), (313, 616))]
            self.video_path = 'data/Footage2.mp4'
            return
            
        if footage_choice == "footage3":
            self.polygones = [[[22, 685], [529, 240], [559, 248], [122, 693]], [[125, 696], [562, 248], [599, 253], [222, 698]], [[327, 694], [607, 259], [648, 259], [433, 696]], [[433, 692], [647, 261], [691, 261], [526, 697]]]
            self.perspective_points = [((522, 252), (692, 265)), ((520, 696), (106, 619))]
            self.video_path = 'data/Footage3.mp4'
            return
    
        if footage_choice == "footage4":
            self.polygones = [[[39, 696], [319, 240], [345, 242], [201, 696]], [[210, 696], [345, 242], [372, 240], [388, 693]], [[632, 698], [411, 247], [438, 247], [793, 696]], [[803, 696], [439, 247], [463, 244], [847, 581]]]
            self.perspective_points = [((319, 241), (455, 237)), ((848, 584), (113, 581))]
            self.video_path = 'data/video2cut.mp4'
            return

        if footage_choice == "footage5":
            self.polygones = [[[39, 696], [319, 240], [345, 242], [201, 696]], [[210, 696], [345, 242], [372, 240], [388, 693]], [[632, 698], [411, 247], [438, 247], [793, 696]], [[803, 696], [439, 247], [463, 244], [847, 581]]]
            self.perspective_points = [((319, 241), (455, 237)), ((848, 584), (113, 581))]
            self.video_path = 'data/videoOvertaking.mp4'
            return













class Main:
    def __init__(self):



        self.video_settings = Video_settings() 

        print("****************************************************** \n", self.video_settings.video_path, "\n**********************************************\n")

        self.model_path = 'data/yolov8n.pt'  
        self.video_path = self.video_settings.video_path  
        self.output_path = 'data/output_video.mp4' 
        self.background = 'data/background.jpeg' 
        self.output_frame_size = (850, 700)  

        self.output_path_img = 'data/output_image.jpeg'
        self.image_path = 'data/background.jpeg'

        using_settings = False

        #self.polygons = [[[39, 696], [319, 240], [345, 242], [201, 696]], [[210, 696], [345, 242], [372, 240], [388, 693]], [[632, 698], [411, 247], [438, 247], [793, 696]], [[803, 696], [439, 247], [463, 244], [847, 581]]]


        # *** DO  NOT DELETE ***
        # clss = Manager()
        # clss.setting(self.output_path_img, self.background, self.video_path, self.output_frame_size)
        # clss.processVideo()
        # self.polygons =  clss.get_polygon_list()

        self.polygons = self.video_settings.polygones

        # print("Polygonelist",self.polygons)

        self.video_processor = VideoProcessor(self.model_path, self.output_path, self.output_frame_size, self.background, self.polygons, self.video_settings.perspective_points)
        self.cap = cv2.VideoCapture(self.video_path)

    def run(self):
        if not self.cap.isOpened():
            print("Error: Video not opened successfully.")
            return

        fps = int(self.cap.get(5))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Use MP4 codec
        out = cv2.VideoWriter(self.output_path, fourcc, fps, self.output_frame_size)


        counter = 0

        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                annotated_frame = self.video_processor.process_frame(frame)
                cv2.imshow("Yolo Tracker", annotated_frame)    # DO NOT DELETE
                
                out.write(annotated_frame)
                
                # DO NOT DELETE #
                # if counter < 20:
                #     cv2.imwrite(f'orignal_image_{counter}.jpeg', frame)
                #     cv2.imwrite(f'orignal_image_{counter}.jpeg', annotated_frame)
                #     counter = counter + 1


                key = cv2.waitKey(1)  # Wait for 1 millisecond
                if key & 0xFF == ord("q"):
                    print("Quitting")
                    break
            else:
                break

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main = Main()
    main.run()