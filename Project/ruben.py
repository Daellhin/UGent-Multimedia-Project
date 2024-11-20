import math
import random
import re
from collections import namedtuple
from typing import Any
import cv2
import numpy as np
import scipy
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
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if cap.isOpened():
        _, frame_3 = cap.read()
        _, frame_2 = cap.read()
        _, frame_1 = cap.read()
        ret, frame = cap.read()
        while cap.isOpened():
            if not ret:
                break
            stacked_frames = np.stack([frame_3,frame_2,frame_1,frame], axis=-1)
            output_frame = np.median(stacked_frames, axis=-1).astype(np.uint8)
            frame_3 = frame_2
            frame_2 = frame_1
            frame_1 = frame
            # Write the processed frame
            out.write(output_frame)

            # Optional: Display the result (comment out for faster processing)
            cv2.imshow('Processing Video', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()


    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()



def main():
    print("Functie")
    process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4", "output/output.mp4")

if __name__ == '__main__':
    main()
