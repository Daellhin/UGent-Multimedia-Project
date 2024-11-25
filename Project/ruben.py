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

def verbeter_frame():
    # Laad de afbeelding
    image = cv2.imread('output/testfoto.png')

    # Converteer de afbeelding naar de YCrCb-kleurruimte
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Splits de kanalen
    y, cr, cb = cv2.split(ycrcb)

    # Pas Gaussian Blur toe op het Y-kanaal om ruis te verminderen
    y_blurred = cv2.GaussianBlur(y, (3, 3), 0)

    # Voeg de kanalen weer samen
    ycrcb = cv2.merge([y_blurred, cr, cb])

    # Converteer terug naar de BGR-kleurruimte
    image_denoised = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Pas histogram equalization toe om het contrast te verbeteren
    image_yuv = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_enhanced = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    # Toon de originele en verbeterde afbeeldingen
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', image_enhanced)

    # Sla de verbeterde afbeelding op
    cv2.imwrite('output/enhanced_image.jpg', image_enhanced)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def verbeter_frame_nieuw():
    # Laad de afbeelding
    image = cv2.imread('output/testfoto.png')

    # Converteer de afbeelding naar de YCrCb-kleurruimte
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Splits de kanalen
    y, cr, cb = cv2.split(ycrcb)

    # Pas Gaussian Blur toe op het Y-kanaal om ruis te verminderen
    y_blurred = cv2.GaussianBlur(y, (5, 5), 0)

    # Voeg de kanalen weer samen
    ycrcb = cv2.merge([y_blurred, cr, cb])

    # Converteer terug naar de BGR-kleurruimte
    image_denoised = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Pas histogram equalization toe om het contrast te verbeteren
    image_yuv = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_enhanced = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    # Verbeter de kleuren
    lab = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Pas een meer geavanceerde ruisonderdrukking toe (Non-Local Means Denoising)
    image_denoised_final = cv2.fastNlMeansDenoisingColored(image_clahe, None, 20, 10, 7, 23)

    # Toon de originele en verbeterde afbeeldingen
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', image_denoised_final)

    # Sla de verbeterde afbeelding op
    cv2.imwrite('output/enhanced_image.jpg', image_denoised_final)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

def stabilize_background(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_bg = cv2.createBackgroundSubtractorMOG2().apply(prev_frame)

    transforms = []
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1):
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_bg = cv2.createBackgroundSubtractorMOG2().apply(curr_frame)

        # Bereken de optische stroom tussen achtergrondframes
        prev_pts = cv2.goodFeaturesToTrack(prev_bg, mask=None, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        good_old = prev_pts[status == 1]
        good_new = curr_pts[status == 1]

        m, _ = cv2.estimateAffinePartial2D(good_old, good_new)
        transforms.append(m)
        prev_gray = curr_gray.copy()
        prev_bg = curr_bg.copy()

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
    print("Achtergrond stabilisatie voltooid!")

def main():
    """print("Functie")
    # Het AVI-bestand dat je wilt converteren
    input_file = "../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4"
    # Het MP4-bestand dat je wilt creëren
    input_mov_file = "output/input_mov_video.mov"
    # Lees het AVI-bestand in
    video_clip = VideoFileClip(input_file)
    # Schrijf het bestand naar MP4 formaat
    video_clip.write_videofile(input_mov_file, codec='libx264')"""

    #stabilize_background("output/input_mov_video.mov", "output/stabilized_video.mp4")

    #stabilizer = VidStab()
    #stabilizer.stabilize(input_path="output/input_mov_video.mov", output_path="output/input_avi_video.avi")


    #process_video("../DegradedVideos/archive_20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4", "output/output.mp4")
    #process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4", "output/tussenstap.mov")

    verbeter_frame_nieuw()


if __name__ == '__main__':
    main()
