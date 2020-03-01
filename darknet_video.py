from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import time
import csv

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def getCoord(pt, matrix):
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    p_after = (int(px), int(py)) # after transformation
    return p_after

def cvDrawBoxes(detections, img, netMain):
    person_count = 0
    chair_count = 0
    table_count = 0
    for detection in detections:
        if detection[0].decode() == "chair" or detection[0].decode() == "person" or detection[0].decode() == "diningtable":
            if detection[0].decode() == "chair": chair_count = chair_count + 1
            if detection[0].decode() == "person": person_count = person_count + 1
            if detection[0].decode() == "diningtable": table_count = table_count +1
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            # pt1 = (xmin, ymin)
            # pt2 = (xmax, ymax)
            # cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            pt1 = (int(round(xmin*1280/netMain)), int(round(ymax*720/netMain)))
            pt2 = (int(round(xmax*1280/netMain)), int(round(ymax*720/netMain)))
            cv2.line(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        [0, 255, 0], 1)
    print("Detected", person_count, "people,", chair_count, "chairs,", table_count, "tables")
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    # out = cv2.VideoWriter(
    #     "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #     (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    print("network width", darknet.network_width(netMain))
    print("network height", darknet.network_height(netMain))
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
                                    

    pts1 = np.float32([[430,200],[880,200],[-230,720],[1480,720]])
    pts2 = np.float32([[0,0],[260,0],[0,426],[260,426]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    
    while True:
        prev_time = time.time()
        # print(time.asctime(time.localtime(prev_time)))
        print(prev_time)
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.15)
        
        image = cvDrawBoxes(detections, frame_rgb, 416)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(format(1/(time.time()-prev_time),'.2f'),"fps")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
    cap.release()
    # out.release()

if __name__ == "__main__":
    YOLO()
