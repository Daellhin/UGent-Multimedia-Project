import math
import random
import re
from collections import namedtuple
from typing import Any
import cv2

import numpy as np
import scipy.ndimage
import simpleaudio as sa
from matplotlib import pyplot as plt
from scipy.io import wavfile
from functools import reduce
from operator import concat

def process_video(input_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Example processing operations:
        
        # 1. Convert to grayscale and back to BGR
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 2. Apply Gaussian blur
        # frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 3. Adjust brightness
        # frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        
        # 4. Edge detection
        # edges = cv2.Canny(frame, 100, 200)
        # frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Write the processed frame
        out.write(frame)
        
        # Optional: Display the result (comment out for faster processing)
        cv2.imshow('Processing Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()