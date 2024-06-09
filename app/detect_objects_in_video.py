import cv2
import numpy as np
import argparse
from ultralytics import YOLO

def detect_objects_in_video(input_video_path, output_video_path, model_path):
    model = YOLO(model_path)

    video = cv2.VideoCapture(input_video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            results = model.predict(source=frame, show=False)

            processed_frame = results[0].plot()
            processed_frame = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)
            out.write(processed_frame)
        else:
            break

    video.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description='Run YOLOv8 object detection on a video.')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('output_video', help='Path to the output video file')
    parser.add_argument('model_path', help='Path to the YOLOv8 model file')
    args = parser.parse_args()

    detect_objects_in_video(args.input_video, args.output_video, args.model_path)

if __name__ == "__main__":
    main()
