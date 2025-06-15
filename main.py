import ultralytics
from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 model
    model = YOLO("yolov11n.pt")  # Load the YOLOv8 Nano model

    # Perform inference on an image
    results = model("")

    # Print the results
    print(results)