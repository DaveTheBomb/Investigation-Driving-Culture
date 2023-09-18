#from typing import Dict, List, Set, Tuple
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

COLORS = sv.ColorPalette.default()

class VideoProcessor:
    def __init__(self, source_weights_path, source_video_path, target_video_path, confidence_threshold = 0.3, iou_threshold = 0.7,):
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2)
        return 

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
        return 

    def annotate_frame(self, frame, detections):
        annotated_frame = frame.copy()
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections, labels)
        return annotated_frame

    def process_frame(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)
        return self.annotate_frame(frame, detections)



def main():
    source_weights_path_ = "data/traffic_analysis.pt"
    source_video_path_ = "data/traffic_analysis.mov"
    target_video_path_ = "data/traffic_analysis_result.mov"
    confidence_threshold_ = 0.3
    iou_threshold_ = 0.7
    processor = VideoProcessor(source_weights_path = source_weights_path_, source_video_path = source_video_path_, target_video_path = target_video_path_, confidence_threshold = confidence_threshold_, iou_threshold = iou_threshold_,)
    processor.process_video()

if __name__ == "__main__":
    main()