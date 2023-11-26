#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
import cv2
import random
import numpy as np

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
