import math
import random
import time
from typing import Sequence
import skimage
from skimage import *
import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.io import wavfile
import moviepy
from visualisations import *
from brent2 import color_adjust, evaluate_frames
from brent2 import ColorParams, Enablers

def getGaussian2D(shape:tuple[2],sigma:float,show=False) -> cv2.typing.MatLike:
    rows, cols = shape
    kernel_x = cv2.getGaussianKernel(cols, sigma)
    kernel_y = cv2.getGaussianKernel(rows, sigma)
    kernel = kernel_y * kernel_x.T
    mask = 1 - kernel / np.linalg.norm(kernel)
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
    if show:
        plt.imshow(mask)
        plt.show()
    return mask

def optimaliseer_kleurrek(frame):
    # Splits de afbeelding in BGR-kanalen
    b, g, r = cv2.split(frame)

    # Definieer het centrum van de afbeelding (assumeer centraal)
    height, width = g.shape
    center_x, center_y = width // 2, height // 2

    def radial_shift_map(shape, scale_factor, power=1.0):
        """Maak een verschuivingskaart op basis van een radiale functie."""
        h, w = shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        dx = x - center_x
        dy = y - center_y
        r = np.sqrt(dx ** 2 + dy ** 2) + 1e-8  # Voeg kleine waarde toe om deling door 0 te vermijden
        # Bereken de verschuiving
        shift_x = scale_factor * (dx / r) * (r ** power)
        shift_y = scale_factor * (dy / r) * (r ** power)
        return shift_x, shift_y

    # Corrigeer een kanaal
    def apply_radial_shift(channel, shift_x, shift_y):
        """Pas de verschuivingen toe op een kleurkanaal."""
        map_x, map_y = np.meshgrid(np.arange(channel.shape[1]), np.arange(channel.shape[0]))
        map_x = (map_x - shift_x).astype(np.float32)
        map_y = (map_y - shift_y).astype(np.float32)
        corrected = cv2.remap(channel, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return corrected

    # Parameters voor verschuivingen
    scale_factor_red = 0.003  # Pas aan op basis van quiver-plot of experimenten
    scale_factor_blue = -0.001

    # Bereken verschuivingskaarten voor rode en blauwe kanalen
    shift_x_r, shift_y_r = radial_shift_map(g.shape, scale_factor_red)
    shift_x_b, shift_y_b = radial_shift_map(g.shape, scale_factor_blue)

    # Corrigeer de rode en blauwe kanalen
    aligned_r = apply_radial_shift(r, shift_x_r, shift_y_r)
    aligned_b = apply_radial_shift(b, shift_x_b, shift_y_b)

    # Combineer de gecorrigeerde kanalen
    aligned_image = cv2.merge((aligned_b, g, aligned_r))

    return aligned_image

def process_video(input_path:str,original:str, output_path:str,color_params:ColorParams, enable:Enablers):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    capOrig = cv2.VideoCapture(original)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {input_path} with {frame_count} frames of {frame_width}x{frame_height} at {fps} frames/second")

    # Create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    eval_frame = 0
    mse_list = []
    psnr_list = []
    ssim_list = []
    while cap.isOpened() and capOrig.isOpened():
        ret, frame = cap.read()
        ret, frameOrig = capOrig.read()
        if not ret:
            break
        if enable.rek:
            frame = optimaliseer_kleurrek(frame)
        frameOut = color_adjust(frame, frameOrig,color_params, enable.show_color_steps)
        if enable.evaluate:
            mse, psnr, ssim = evaluate_frames(frame,frameOrig)
            mse_list.append(mse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        out.write(frameOut)
        eval_frame+=1

        if enable.show_processed_frame:
            cv2.imshow('Processing Video', frameOut)
        if enable.show_color_steps or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if mse_list:
        print("MSE=",np.mean(mse_list)," PSNR=",np.mean(psnr_list)," SSIM=",np.mean(ssim_list))
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    print(moviepy.__version__)
    """video_clip = VideoFileClip("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4")
    W,H = video_clip.size
    print(W,H)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile("output/original_audio.mp3")
    audio_clip = AudioFileClip("output/edit_audio.mp3")
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile("output/original_video.mp4")
    audio_clip.close()
    video_clip.close()"""

    timestamp = time.strftime("%d-%m-%Y_%H%M%S")
    noEffectColor = ColorParams()
    obamaColor = ColorParams(3, 1080, 0.005, 0.005, 1/3, 2.5, 2.1, 0, 190, 140, 20, 1, 0, 1)
    allOff = Enablers(show_processed_frame=True)
    edit_no_show = Enablers(rek=True,show_processed_frame=True,evaluate=True)
    """process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "../SourceVideos/2017-01-07_President_Obama's_Weekly_Address.mp4",
                  f"output/2017-01-07_President_Obama's_Weekly_Address_{timestamp}.mp4",
                  obamaColor, edit_no_show)
    femaleColor = ColorParams(3, 1080, 0, 0, 1/2, 2.5, 2.1, 30, 190, 140, 40, 1, 0, 1)
    process_video("../DegradedVideos/archive_20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "../SourceVideos/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  f"output/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows_{timestamp}.mp4",
                  femaleColor, edit_no_show)
    process_video("../DegradedVideos/archive_Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
                  "../SourceVideos/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
                  f"output/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie_{timestamp}.mp4",
                  obamaColor, edit_no_show)
    femaleColor = ColorParams(3, 1080, 0, 0, 1 / 2, 2.5, 2.1, 30, 190, 140, 50, 1, 10, 1)
    process_video("../DegradedVideos/archive_Robin_Singing_video.mp4",
                  "../SourceVideos/Robin_Singing_video.mp4",
                  f"output/Robin_Singing_video_{timestamp}.mp4",
                  femaleColor, edit_no_show)
    process_video("../DegradedVideos/archive_Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
                  "../SourceVideos/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
                  f"output/Jasmine_Rae_-_Heartbeat_(Official_Music_Video)_{timestamp}.mp4",
                  femaleColor, edit_no_show)
    process_video("../ArchiveVideos/Apollo_11_Landing_-_first_steps_on_the_moon.mp4",
                  "../ArchiveVideos/Apollo_11_Landing_-_first_steps_on_the_moon.mp4",
                  f"output/Apollo_11_Landing_-_first_steps_on_the_moon_{timestamp}.mp4",
                  noEffectColor,allOff)"""
    archive = ColorParams(3, 1080//2, 0, 0, 2/3, 1, 1, 5, 0, 0, -20, 1, 5, 1)
    process_video("..\ArchiveVideos\Breakfast-at-tiffany-s-official®-trailer-hd.mp4",
                  "..\ArchiveVideos\Breakfast-at-tiffany-s-official®-trailer-hd.mp4",
                  f"output\Breakfast-at-tiffany-s-official®-trailer-hd_{timestamp}.mp4",
                  archive,allOff)
    """archive = ColorParams(3, 1080, 0, 0, 1 / 2, 1, 1, 30, 0, 0, -15, 1, -5, 1)
    process_video("..\ArchiveVideos\President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962.mp4",
                  "..\ArchiveVideos\President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962.mp4",
                  f"output\President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962_{timestamp}.mp4",
                  archive,allOff)
    process_video("..\ArchiveVideos\Edison_speech,_1920s.mp4",
                  "..\ArchiveVideos\Edison_speech,_1920s.mp4",
                  f"output\Edison_speech,_1920s_{timestamp}.mp4",
                  noEffectColor, allOff)
    process_video("..\ArchiveVideos\The_Dream_of_Kings.mp4",
                  "..\ArchiveVideos\The_Dream_of_Kings.mp4",
                  f"output\The_Dream_of_Kings_{timestamp}.mp4",
                  noEffectColor,allOff)"""

if __name__ == '__main__':
    main()