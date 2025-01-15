from inference.models.yolo_world.yolo_world import YOLOWorld
import supervision as sv
from tqdm import tqdm

class YOLO():
    def __init__(self, model_id="yolo_world/l", confidence=0.01, nms_threshold=0.1):
        self.model = YOLOWorld(model_id=model_id)
        self.confidence = confidence
        self.nms_threshold = nms_threshold

    def process_video(self, classes, source_video_path):
        self.model.set_classes(classes)

        frame_generator = sv.get_video_frames_generator(source_video_path)
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        width, height = video_info.resolution_wh
        frame_area = width * height

        detections_list = []
        for frame in tqdm(frame_generator, total=video_info.total_frames, desc="YOLO-World"):
            results = self.model.infer(frame, confidence=self.confidence)
            detections = sv.Detections.from_inference(results).with_nms(threshold=self.nms_threshold)
            detections = detections[(detections.area / frame_area) < 0.1]
            detections_list.append(detections)
        return detections_list
