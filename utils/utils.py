from datetime import timedelta
import supervision as sv
from PIL import Image
import time
import cv2
import re
import os
os.makedirs('tmp', exist_ok=True)

last_time = 0

def tic():
    global last_time
    last_time = time.time()

def toc():
    global last_time
    diff = time.time() - last_time
    last_time = time.time()
    return diff

pattern = r"<<(\d{2}:\d{2}),(\d{2}:\d{2})>>(?:\s*:\s*(.*))?"
def extract_timeframe(text):
    match = re.search(pattern, text)
    if match:
        start_time = match.group(1)
        end_time = match.group(2)
        # Check if the description exists (it may be None if not present)
        description = match.group(3) if match.group(3) is not None else ""
        result = ((start_time, end_time), description)
        return result

def classify_content(content_path):
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    image_exts = ['.png', '.jpg', '.jpeg', '.gif']
    ext = os.path.splitext(content_path)[1].lower()
    ctype = 'video' if ext in video_exts else 'image' if ext in image_exts else None
    assert ctype, f"Unsupported content type: {ext}"
    return ctype

def parse_time(time_str):
    parts = [0, 0, 0]  # Default values for h, m, s
    matches = re.findall(r'\d+', time_str)
    matches = [int(x) for x in matches[-3:]]
    parts[-len(matches):] = matches
    
    return tuple(parts)

def extract_frame_from_video(video_path, time_str):
    """
    Extract a frame from the video at the given timestamp (time_str: '00:00:10' for 10th second).
    Returns a PIL image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    h, m, s = parse_time(time_str)
    frame_time = max((h * 3600 + m * 60 + s) * 1000, 200)
    
    # Set the video to the frame at the specific time
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
    success, frame = cap.read()

    if not success:
        raise ValueError(f"Could not extract frame at {time_str}")

    cap.release()

    frame_path = f"./tmp/frame_{frame_time}.png"
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame.save(frame_path)
    return frame_path
# ...

def frame_to_time(frame, fps, include_ms=False):
    seconds = frame / fps
    if include_ms:
        return str(timedelta(seconds=seconds))
    else:
        return str(timedelta(seconds=seconds)).split(".")[0]

def merge_intervals(intervals, merge_threshold_ms, fps):
    merged = []
    merge_threshold_sec = merge_threshold_ms / 1000
    intervals.sort(key=lambda x: x[0])
    for start, end in intervals:
        if not merged:
            merged.append((start, end))
        else:
            last_start, last_end = merged[-1]
            time_diff = (start - last_end) / fps
            if time_diff <= merge_threshold_sec:
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))
    return merged

def get_object_intervals(classes, detections, source_video_path, merge_threshold_ms=1500):
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    object_intervals = {cls: [] for cls in classes}
    last_frames = {cls: None for cls in classes}

    for frame, detection in enumerate(detections):
        if "class_name" in detection.data and len(detection.data["class_name"]) > 0:
            class_name = detection.data['class_name'][0]
            if last_frames[class_name] is not None and frame == last_frames[class_name] + 1:
                start_frame, _ = object_intervals[class_name][-1]
                object_intervals[class_name][-1] = (start_frame, frame)
            else:
                object_intervals[class_name].append((frame, frame))
            last_frames[class_name] = frame
        else:
            for cls in last_frames:
                if last_frames[cls] is not None:
                    start_frame, end_frame = object_intervals[cls][-1]
                    if end_frame == last_frames[cls]:
                        object_intervals[cls][-1] = (start_frame, frame - 1)
                    last_frames[cls] = None

    for cls in object_intervals:
        intervals = object_intervals[cls]
        object_intervals[cls] = merge_intervals(intervals, merge_threshold_ms, video_info.fps)

    result = {}
    for cls, intervals in object_intervals.items():
        time_intervals = [
            (frame_to_time(start, video_info.fps), frame_to_time(end, video_info.fps))
            for start, end in intervals
        ]
        result[cls] = time_intervals

    return result

def save_detections_video(detections_list, source_video_path, target_video_path):
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK)
    frame_generator = sv.get_video_frames_generator(source_video_path)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame, detections in zip(frame_generator, detections_list):
            annotated_frame = frame.copy()
            annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections)
            sink.write_frame(annotated_frame)

def CustomException(name, message):
    exception_class = type(name, (Exception,), {'__init__': lambda self, msg=message: setattr(self, 'message', msg)})
    return exception_class(message)

def get_video_duration(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    duration_in_seconds = int(total_frames / fps)
    minutes = duration_in_seconds // 60
    seconds = duration_in_seconds % 60
    duration_str = f"{minutes}:{seconds:02d}"
    
    video.release()
    return duration_str

def trim_video(video_path, time_range):
    """
    Trim a video to the specified time range and save it to a temporary file.
    The range is extended to 1 second after the end second, or up to the end of the video.
    
    Args:
        video_path (str): Path to the input video file.
        time_range (str): Time range in 'MM:SS,MM:SS' format.
    
    Returns:
        str: Path to the saved trimmed video file.
    """
    # Parse the time range
    start_time, end_time = time_range.split(',')
    start_seconds = int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])
    end_seconds = int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1]) + 2.5
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = total_frames // fps
    
    # Cap the end time to the video duration
    end_seconds = min(end_seconds, video_duration_seconds)
    
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    
    # Generate the output path based on the input path and time range
    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_{start_time.replace(':', '')}_{end_time.replace(':', '')}{ext}"
    
    # Set the starting frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Read and write frames within the range
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = video.read()
        if not ret:
            break
        output.write(frame)
        current_frame += 1
    
    # Release resources
    video.release()
    output.release()
    
    return output_path
