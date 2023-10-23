1. Create a virtual environment
2. Clone the repository:
    git@github.com:DaveTheBomb/Investigation-Driving-Culture.git 
3. Go to cloned folder :
    cd YOLOv8-DeepSORT-Object-Tracking
4. Open the termal and install dependencies
5.Open the terminal and set the directory 
   cd ultralytics/yolo/v8/detect
6. Put the video to be detected in ultralytics/yolo/v8/detect
7 Run the code on the terminal using the following command:
   python predict.py model=yolov8n.pt source="test3.mp4" show=True 
   Other pretrained weights: yolov8s.pt, yolov8m.pt, yolov8l.pt
                             yolov8x.pt