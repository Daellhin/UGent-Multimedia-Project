import math
import random
import cv2
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from scipy.io import wavfile
from moviepy.editor import *

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

def process_video(input_path, output_path,s=2,k=0.1):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    mask = gaussian_filter((frame_height,frame_width), s)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the Wiener filter to each color channel
        output_0 = wiener_filter(frame[:, :, 0], mask, k)
        output_1 = wiener_filter(frame[:, :, 1], mask, k)
        output_2 = wiener_filter(frame[:, :, 2], mask, k)

        output_0 = cv2.normalize(output_0,None,0,255,norm_type=cv2.NORM_MINMAX)
        output_1 = cv2.normalize(output_1, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        output_2 = cv2.normalize(output_2, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        #output_0 = np.clip(output_0, 0, 255)
        #output_1 = np.clip(output_1, 0, 255)
        #output_2 = np.clip(output_2, 0, 255)

        # Merge the color channels back together
        frame = cv2.merge((output_0, output_1, output_2)).astype(np.uint8)

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

def main():
    video_clip = VideoFileClip("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4")
    W,H = video_clip.size
    print(W,H)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile("output/original_audio.mp3")
    audio_clip = AudioFileClip("output/edit_audio.mp3")
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile("output/original_video.mp4")
    audio_clip.close()
    video_clip.close()

    #process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4", "output/output.mp4",1,0.4)
    #process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4", "output/output1.mp4",1.5,0.5)
    #process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4", "output/output2.mp4",2,0.6)

if __name__ == '__main__':
    main()