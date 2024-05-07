import numpy as np
import os
import cv2
import csv

def read_data(folder_path):
    """ Reads data and returs dictionares of calib and ground truth labels """
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