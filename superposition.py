# Project: Human Pose Estimation Using Deep Learning in OpenCV
# Author: Addison Sears-Collins
# Date created: February 25, 2021
# Description: A program that takes a video with a human as input and outputs
# an annotated version of the video with the human's position and orientation..
 
# Reference: https://github.com/quanhua92/human-pose-estimation-opencv
 
# Import the important libraries
import cv2 as cv # Computer vision library
import numpy as np # Scientific computing library
 
# Make sure the video file is in the same directory as your code
filename = 'dancing32.mp4'
file_size = (1920,1080) # Assumes 1920x1080 mp4 as the input video file
 
# We want to save the output to a video file
output_filename = 'dancing32_output.mp4'
output_frames_per_second = 20.0
 
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
 
POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
 
# Width and height of training set
inWidth = 368
inHeight = 368
 
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
 
cap = cv.VideoCapture(filename)
 
# Create a VideoWriter object so we can save the video output
fourcc = cv.VideoWriter_fourcc(*'mp4v')
result = cv.VideoWriter(output_filename,  
                         fourcc, 
                         output_frames_per_second, 
                         file_size) 
# Process the video
while cap.isOpened():
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
 
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
     
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
 
    assert(len(BODY_PARTS) == out.shape[1])
 
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]
 
        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        # Feel free to adjust this confidence value.  
        points.append((int(x), int(y)) if conf > 0.2 else None)
 
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)
 
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
 
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (255, 0, 0), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (255, 0, 0), cv.FILLED)
 
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
 
    # Write the frame to the output video file
    result.write(frame)
         
# Stop when the video is finished
cap.release()
     
# Release the video recording
result.release()