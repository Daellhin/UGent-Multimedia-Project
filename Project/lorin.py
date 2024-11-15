import math
import random
import re
from collections import namedtuple
from typing import Any
import cv2
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from scipy.io import wavfile
from functools import reduce
from operator import concat
import os

from utils import printProgressBar


def process_video(input_path, output_path):
    print("Processing: ", input_path)
    cap = cv2.VideoCapture(input_path)

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    printProgressBar(0, total_frames, length=50)
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # -- Start video processing functions --

        # -- End video processing functions --

        cv2.imshow('Processing Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame)
        printProgressBar(index + 1, total_frames, length=50)
        index += 1

    # Release
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    process_video(
        "DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
        "output/output.mp4",
    )


if __name__ == "__main__":
    main()
