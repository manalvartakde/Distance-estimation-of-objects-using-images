import numpy as np
import os
import sys
import cv2
import csv
import torch
from PIL import Image
from pathlib import Path

def read_data(folder_path):
    """Args: Folder path 
    Returns: data in the form of dictionares of calib and ground truth labels """
    calib = {}
    labels = {}
    folders = ["calib", "labels"]
    for folder in folders:
        pth = folder_path + folder
        f = os.listdir(pth) # lists all the csv files in folder
        for file in f:                              # loops tyhrough all csv files
            pth = folder_path + folder + "/" + file  # complete csv file location
            arr = []
            if(folder == "labels"):
                file1 = open(pth)
                csvreader = csv.reader(file1)
                for row in csvreader:
                    # print(row)
                    arr.append("".join(row).split())
                labels[file[:6]] = arr
                continue
            if(folder == "calib"):
                arr = np.loadtxt(pth, delimiter=" ")
                calib[file[:6]] = np.array(arr, dtype=np.float16)
    return(labels, calib)



def detect_and_draw(image_path):
    """Args: Image path
    Detects cars in the image 
    Returns image mat with bounding box around the cars and array with corner coordinates of bounding rectangle"""
    # YOLO v5 library path
    yolov5_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'ultralytics_yolov5_master'
    sys.path.append(str(yolov5_dir))

    # Load YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    # Just makes sure that the image in RGB
    img = Image.open(image_path).convert("RGB")
    # model function used the pretrained yolo model to detect OBJECTS in the image
    results = model(img, size=640) # Image is resized to 640 x 640

    # Drawing bounding boxes around car
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    car_bounding_boxes = []
    for box in results.xyxy[0]:
        label = int(box[-1])
        if label == 2:  # here class value 2 is for cars..... we elimainte other detected objects
            xmin, ymin, xmax, ymax = map(int, box[:4])
            car_bounding_boxes.append([xmin, ymin, xmax, ymax])
            cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # Display Images
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return(image_rgb, car_bounding_boxes)


def augument_pixel(pixel_val_arr):
    """Args: Array of all pixel coordinates[[x1, y1], [x2, y2]]
    Returns: Augumented array [[x1, y1, 1], [x2, y2, 1]]"""
    for pixel_val in pixel_val_arr:
        for coordinate in pixel_val:
            coordinate.append(1)
    return(np.array(pixel_val_arr))

def inv_proj(pixel_val_arr, const = 3):
    """Converts 3x1 Augumented vector into 
       Args: Array with augumented pixel coordinates, constant to be multiplied
       Returns: Helper coordinates (Pixel coordinates multiplied by a constant /// Represented by z in slides)"""
    return(np.array(pixel_val_arr * const))