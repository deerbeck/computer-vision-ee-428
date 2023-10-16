import cv2

from magicwand import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('video',help='path to input video file')
parser.add_argument('--output',help='path to output video file (optional)')
parser.add_argument('--calibration',default='iphone_calib.txt',help='path to calibration file')
parser.add_argument('--ball_radius',type=float,default=3,help='radius of ball in cm')
args = parser.parse_args()

wand = MagicWand(calibration_path=args.calibration,R=args.ball_radius)

cap = cv2.VideoCapture(args.video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    wand.process_frame(frame)
     
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break

