import math
import os
import random
import re
from collections import namedtuple
from functools import reduce
from operator import concat
from typing import Any, Callable
from utils import printProgressBar
import cv2
import numpy as np
import scipy.ndimage
import simpleaudio as sa
from matplotlib import pyplot as plt
from moviepy.editor import AudioClip, VideoFileClip, AudioFileClip
from scipy.io import wavfile
from scipy.signal import istft, stft, wiener

show_images = True

def play_stereo_audio(stereo_audio):
    # Create a 2-channel stereo audio buffer
    sample_rate = 44100
    
    # Play the stereo audio buffer
    play_obj = sa.play_buffer(stereo_audio, 2, 2, sample_rate)
    play_obj.wait_done()


def create_audio_clip(samples, fps):
    def make_frame(t):
        return np.array(samples)
       
    return AudioClip(make_frame, duration=len(samples) / fps)

def reduce_noise(
    audio_samples: list[list[float]], filter_size: int = 5
) -> list[list[float]]:
    """
    Apply Wiener noise reduction to audio samples

    Args:
        audio_samples: List of audio sample frames
        filter_size: Size of the Wiener filter window (default 5)

    Returns:
        Noise-reduced audio samples
    """
    # Convert to numpy array for processing
    audio_array = np.array(audio_samples)

    # Apply Wiener filter along each channel
    denoised_array = np.array(
        [wiener(channel, filter_size) for channel in audio_array.T]
    ).T

    return denoised_array.tolist()


def process_frame(
    get_frame: Callable[[float], np.ndarray], time: float, frame_duration: float
):
    """
    Process frame callback:
    - use get_frame with time to get current frame
    - use get_frame with multiple times to get multiple frames(much slower)
    """
    # -- Get frame(s) --
    frame = get_frame(time)
    # times = np.arange(0, 4) * frame_duration
    # frames = list(map(get_frame, times))

    # -- Modify frame --

    # -- Debug images --
    global show_images
    if show_images:
        im_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Frame", im_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        show_images = not show_images

    return frame


def process_video(input_path: str, output_path: str):
    print("Processing: ", input_path)

    video = VideoFileClip(input_path)
    # -- Video processing --
    # frame_duration = 1 / video.fps  # frame duration in seconds
    # processed_video: VideoFileClip = video.fl(
    #     lambda gf, t: process_frame(gf, t, frame_duration)
    # )
    # cv2.destroyAllWindows()
    processed_video = video  # for debuging audio without video

    # -- Audio processing --
    fs = video.fps
    audio_samples: list[list[float]] = list(video.audio.iter_frames())
    processed_audio = reduce_noise(audio_samples)

    wavfile.write(f"{output_path}.wav", 4400, np.array(processed_audio, dtype=np.float32))
    t = AudioFileClip("output/output.mp4.wav")
    result: VideoFileClip = processed_video.set_audio(t)
    #play_stereo_audio(processed_audio)

    # -- Output results --
    result.write_videofile(output_path)


def main():
    process_video(
        "DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
        "output/output.mp4",
    )


if __name__ == "__main__":
    main()
