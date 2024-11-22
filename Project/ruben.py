import math
import random
import re
from collections import namedtuple
from typing import Any
import cv2
import numpy
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.io import wavfile
from functools import reduce
from operator import concat
import moviepy
from vidstab import VidStab
from moviepy import VideoFileClip

def avg_filter(frame):
    kleur0 = frame[:, :, 0]
    kleur1 = frame[:, :, 1]
    kleur2 = frame[:, :, 2]

    H = gaussian_filter(kleur0.shape,sigma=2)

    window_size = (3,3)

    output_0 = scipy.ndimage.median_filter(kleur0,size=window_size)
    output_0 = wiener_filter(output_0,H,0.4)
    output_0 = numpy.clip(output_0,0,248)
    output_0 = cv2.normalize(output_0,None,0,255,norm_type=cv2.NORM_MINMAX)

    output_1 = scipy.ndimage.median_filter(kleur1,size=window_size)
    output_1 = wiener_filter(output_1,H,0.4)
    output_1 = numpy.clip(output_1,0,248)
    output_1 = cv2.normalize(output_1, None, 0, 255, norm_type=cv2.NORM_MINMAX)

    output_2 = scipy.ndimage.median_filter(kleur2,size=window_size)
    output_2 = wiener_filter(output_2,H,0.4)
    output_2 = numpy.clip(output_2,0,248)
    output_2 = cv2.normalize(output_2, None, 0, 255, norm_type=cv2.NORM_MINMAX)

    output = cv2.merge((output_0, output_1, output_2)).astype(np.uint8)
    return output

def gaussian_filter(shape, sigma):
    """Genereer een Gaussiaans filtermasker met de opgegeven afbeeldingsgrootte en sigma."""
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2

    # Maak een grid van (x, y)-coördinaten met het centrum op de middenpositie
    y, x = np.ogrid[:rows, :cols]
    y -= center_row
    x -= center_col

    # Bereken het Gaussische filtermasker
    H = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    H /= np.sum(H)  # Normeer zodat de som gelijk is aan 1
    return H

def wiener_filter(image, H, k):
    H = np.roll(H,int(H.shape[0]/2),0)
    H = np.roll(H, int(H.shape[1] / 2), 1)

    # frequency of modified filter
    H_fft = np.fft.fft2(H,s=image.shape)
    H_fft = np.fft.fftshift(H_fft)

    # Compute the Wiener filter
    F = np.conj(H_fft) / (np.abs(H_fft) ** 2 + k)

    # Compute the FFT of the image
    image_fft = np.fft.fft2(image)
    image_fft_shifted = np.fft.fftshift(image_fft)

    image_f_filter = image_fft_shifted * F

    ifshift = np.fft.ifftshift(image_f_filter)
    ifbeeld = np.fft.ifft2(ifshift)
    return ifbeeld.real

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
            # Mediaan berekenen
            stacked_frames = np.stack([frame_3,frame_2,frame_1,frame], axis=-1)
            output_frame = np.median(stacked_frames, axis=-1).astype(np.uint8)

            output_frame = avg_filter(output_frame)
            # Write the processed frame
            out.write(output_frame)

            # Optional: Display the result (comment out for faster processing)
            cv2.imshow('Processing Video', output_frame)

            frame_3 = frame_2
            frame_2 = frame_1
            frame_1 = frame

            ret, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def stabilize_video(input_path, output_path):
    # Open het video bestand
    cap = cv2.VideoCapture(input_path)
    # Haal de frames per seconde (fps) op
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Haal de breedte en hoogte van de frames op
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Definieer de codec en maak VideoWriter object
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initiële feature points detectie met ShiTomasi corner detector
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    transforms = []
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1):
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
        # Filter only good points
        good_old = prev_pts[status == 1]
        good_new = curr_pts[status == 1]
        # Estimate transformation matrix
        m = cv2.estimateAffinePartial2D(good_old, good_new)[0]
        transforms.append(m)
        prev_gray = curr_gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

    # Apply transformation to stabilize video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(len(transforms)):
        ret, frame = cap.read()
        if not ret:
            break
        m = transforms[i]
        stabilized_frame = cv2.warpAffine(frame, m, (width, height))
        out.write(stabilized_frame)

    cap.release()
    out.release()
    print("Video stabilisatie voltooid!")


def main():
    print("Functie")
    # Het AVI-bestand dat je wilt converteren
    input_file = "../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4"
    # Het MP4-bestand dat je wilt creëren
    input_mov_file = "output/input_mov_video.mov"
    # Lees het AVI-bestand in
    video_clip = VideoFileClip(input_file)
    # Schrijf het bestand naar MP4 formaat
    video_clip.write_videofile(input_mov_file, codec='libx264')

    stabilize_video("output/input_mov_video.mov", "output/stabilized_video.mp4")

    #stabilizer = VidStab()
    #stabilizer.stabilize(input_path="output/input_mov_video.mov", output_path="output/input_avi_video.avi")


    #process_video("../DegradedVideos/archive_20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4", "output/output.mp4")
    #process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4", "output/tussenstap.mov")




if __name__ == '__main__':
    main()
