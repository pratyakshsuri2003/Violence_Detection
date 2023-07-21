import io
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
from flask_cors import CORS
import pandas as pd
import os
import argparse
import logging
import cv2
import numpy as np
import base64
import tempfile

df = pd.DataFrame()
df["xmin"] = [50]
df["ymin"] = [50]
df["xmax"] = [50]
df["ymax"] = [50]
df["confidence"] = [0.3]
df["class"] = [4]
df["name"] = [' ']

model = torch.hub.load("F:/Workspace/YOLO V5/yolov5/", 'custom',
                       path=r"F:\Workspace\Model PT Files\pratyaksh_yolov5_5\best.pt",
                       force_reload=True, source='local')

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


@app.route("/test", methods=["GET"])
def check_api():
    return jsonify({"status": 200})


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/detection", methods=["POST"])
def detect():
    app.logger.info("entered method")
    if not request.method == "POST":
        return jsonify({"error": "Invalid request method"})

    if "image" not in request.files and "video" not in request.files:
        return jsonify({"error": "No image or video uploaded"})

    if "video" in request.files:
        video_file = request.files["video"]
        return process_video(video_file)

    image_file = request.files["image"]
    image_bytes = image_file.read()
    img = Image.open(io.BytesIO(image_bytes))

    try:
        results = model(img, size=640)  # reduce to 320 for faster inference
        detection = results.pandas().xyxy[0]

        # Convert detection results to a list of dictionaries
        detection_list = detection.to_dict(orient="records")

        # Draw bounding boxes on the image using OpenCV
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for detection in detection_list:
            xmin, ymin, xmax, ymax = int(detection["xmin"]), int(detection["ymin"]), int(detection["xmax"]), int(
                detection["ymax"])
            class_name = mapClassToName(detection["class"])
            confidence = detection["confidence"]
            cv2.rectangle(img_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img_cv2, f"{class_name}: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        # Convert the modified image back to PIL format
        img_with_bboxes = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

        # Encode the image to base64 and return it in the response
        buffered = io.BytesIO()
        img_with_bboxes.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = {
            "type": "image",
            "detection": detection_list,
            "data": img_base64
        }

        app.logger.info("detected")
        return jsonify(response)
    except Exception as e:
        app.logger.error(f'Error {e}')
        return jsonify({"error": "An error occurred during detection"})


def mapClassToName(classValue):
    class_names = ["Gun", "Knife", "Military_Vehicles", "Terrorist_Flags", "Unknown"]
    return class_names[classValue] if 0 <= classValue < len(class_names) else "Unknown"


# Import timedelta from datetime
from datetime import timedelta

def process_video(video_file):
    try:
        # Save the video file to a temporary directory
        temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        video_file.save(temp_video_path)

        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_duration = timedelta(seconds=total_frames / fps)

        if video_duration.total_seconds() != 30:
            os.remove(temp_video_path)
            return jsonify({"error": "Video must be exactly 30 seconds long"})

        # Rest of the code for processing the video goes here...
        # (same as before)

        response = {
            "type": "video",
            "results": results_list
        }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f'Error {e}')
        return jsonify({"error": "An error occurred during detection"})


def process_video(video_file):
    try:
        # Save the video file to a temporary directory
        temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        video_file.save(temp_video_path)

        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        chunk_size = 30  # Number of frames to process at a time (adjust this based on memory constraints)

        results_list = []
        for i in range(0, total_frames, chunk_size):
            frames = []
            for _ in range(chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            detections_batch = []
            for frame in frames:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = model(img_pil, size=640)
                detection = results.pandas().xyxy[0]
                detection_list = detection.to_dict(orient="records")
                detections_batch.append(detection_list)

            for frame, detection_list in zip(frames, detections_batch):
                # Draw bounding boxes on the image using OpenCV
                img_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for detection in detection_list:
                    xmin, ymin, xmax, ymax = int(detection["xmin"]), int(detection["ymin"]), int(
                        detection["xmax"]), int(detection["ymax"])
                    class_name = mapClassToName(detection["class"])
                    confidence = detection["confidence"]
                    cv2.rectangle(img_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(img_cv2, f"{class_name}: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                # Convert the modified image back to PIL format
                img_with_bboxes = Image.fromarray(img_cv2)

                # Save the modified frame with bounding boxes
                result_frame_path = os.path.join(tempfile.gettempdir(), f"result_frame_{i}.jpg")
                img_with_bboxes.save(result_frame_path, format="JPEG")

                # Encode the image to base64 and store in results_list
                with open(result_frame_path, "rb") as frame_file:
                    img_base64 = base64.b64encode(frame_file.read()).decode("utf-8")
                    results_list.append({
                        "detection": detection_list,
                        "data": img_base64
                    })

        # Delete the temporary video file
        os.remove(temp_video_path)

        response = {
            "type": "video",
            "results": results_list
        }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f'Error {e}')
        return jsonify({"error": "An error occurred during detection"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=2003, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
