import math
import random
from typing import Sequence

import cv2
import numpy as np
import scipy
from PIL.ImageChops import multiply
from matplotlib import pyplot as plt
from scipy.io import wavfile
import moviepy
from visualisations import *

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


def process_frame(frame:cv2.typing.MatLike, frameOrig:cv2.typing.MatLike,show_steps=False,evaluate=False,** kwargs) -> tuple[cv2.typing.MatLike, list[float], list[float], list[float]]:
    """Verwerkt frame, mogelijke tweaks:
        params = {"YfilterSize": 5, "sigma": 1080/2,"gausUVAdj": 0.01,"gausYAdj": 0.5,"Umultiply": 2.5,"Usubstract": 75,"Vmultiply": 2.1,"Vsubstract": 70,"Sadd": 20,"Smultiply": 1}"""
    params = {  # Standaardwaarden
    "YfilterSize": 5,
    "sigma": 1080 / 2,
    "gausUVAdj": 0.01,
    "gausYAdj": 0.5,
    "Umultiply": 2.5,
    "Usubstract": 75,
    "Vmultiply": 2.1,
    "Vsubstract": 70,
    "Sadd": 20,
    "Smultiply": 1
    }
    params.update(kwargs)

    # YUV modifier - kringverzwakking
    yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    yuv_or = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YUV)
    yo, uo, vo = cv2.split(yuv_or)

    if show_steps:
        cv2.imwrite("output/start_Frame.jpg",frame)
        show_histogram(y, yo, "Y", "Y Original")
        #show_spectrum(y, yo, "Y")
        show_histogram(u, uo, "U", "U original")
        #show_spectrum(u, uo, "U")
        show_histogram(v, vo, "V", "V Original")
        #show_spectrum(v, vo, "V")

    y = scipy.ndimage.median_filter(y, (params["YfilterSize"],params["YfilterSize"]))
    mask = getGaussian2D(y.shape,params["sigma"],show_steps)
    y = cv2.multiply(y.astype(np.float64), (1-params["gausYAdj"]/2)+params["gausYAdj"]*mask)
    u = cv2.multiply(u.astype(np.float64), 1+params["gausUVAdj"]*mask)
    v = cv2.multiply(v.astype(np.float64), 1+params["gausUVAdj"]*mask)
    u = cv2.subtract(u, params["Usubstract"])
    u = cv2.multiply(u,params["Umultiply"])
    v = cv2.subtract(v, params["Vsubstract"])
    v = cv2.multiply(v,params["Vmultiply"])
    y = np.clip(y,0,255).astype(np.uint8)
    u = np.clip(u,0,255).astype(np.uint8)
    v = np.clip(v,0,255).astype(np.uint8)
    frame = cv2.merge((y, u, v))
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    if show_steps:
        cv2.imwrite("output/YUV-edit_Frame.jpg",frame)
        show_histogram(y, yo, "Y edit", "Y Original")
        show_histogram(u, uo, "U edit", "U original")
        show_histogram(v, vo, "V edit", "V Original")

    # BGR modifier
    b, g, r = cv2.split(frame)
    bo, go, ro = cv2.split(frameOrig)
    #r = cv2.multiply(r,0.80)
    #g = cv2.multiply(g,0.70)
    #b = cv2.multiply(b,0.6)
    #r = np.clip(r,0,255).astype(np.uint8)
    #g = np.clip(g, 0, 255).astype(np.uint8)
    #b = np.clip(b, 0, 255).astype(np.uint8)
    #frame = cv2.merge((b,g,r))
    if show_steps:
        cv2.imwrite("output/BGR-edit_frame.jpg",frame)
        show_histogram(r, ro, "Red After", "Red Original")
        show_histogram(g, go, "Green After", "Green Original")
        show_histogram(b, bo, "Blue After", "Blue Original")

    #HSV modifiers
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsvOrig = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2HSV)
    ho, so, vo = cv2.split(hsvOrig)

    #h = cv2.multiply(h,0.99)
    #v = cv2.multiply(v, 0.90)
    s = cv2.multiply(s,params["Smultiply"])
    s = cv2.add(s, params["Sadd"])
    h = np.clip(h,0,255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    s = np.clip(s, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    if show_steps:
        show_histogram(h, ho, "Hue After", "Hue Original")
        show_histogram(v, vo, "Value After", "Value Original")
        show_histogram(s, so, "Saturation After", "Saturation Original")

    if evaluate:
        mae = cv2.mean(np.abs(cv2.subtract(frame, frameOrig)))
        mse = cv2.mean(cv2.pow(cv2.subtract(frame, frameOrig), 2))
        psnr = (10 * cv2.log(cv2.divide(255 ** 2, mse))).T[0]
        print(f"MAE: B={mae[0]:.3f} G={mae[1]:.3f} R={mae[2]:.3f}\tMSE: B={mse[0]:.3f} G={mse[1]:.3f} R={mse[2]:.3f}\tPSNR: B={psnr[0]:.3f} G={psnr[1]:.3f} R={psnr[2]:.3f}")
        return frame, list(mae[:3]), list(mse[:3]), list(psnr[:3])
    return frame, [-1,-1,-1],[-1,-1,-1],[-1,-1,-1]

def process_frame2(frame: cv2.typing.MatLike, frameOrig: cv2.typing.MatLike, show_steps=False, evaluate=False,
                   params=None) -> tuple[cv2.typing.MatLike, list[float], list[float], list[float]]:
    """Verwerkt frame, mogelijke tweaks:
        params = {"filterSize","sigma","gaus<x>Adj","<x>multiply","<x>substract","Sadd""Smultiply"}"""
    if params is None:
        params = {
            "filterSize": 5,
            "sigma": 1080 / 2,
            "gausRAdj": 0.01,
            "gausGAdj": 0.01,
            "gausBAdj": 0.01,
            "Rmultiply": 1,
            "Rsubstract": 0,
            "Gmultiply": 1,
            "Gsubstract": 0,
            "Bmultiply": 1,
            "Bsubstract": 0,
            "Sadd": 0,
            "Smultiply": 1
        }

    # HSV modifiers
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsvOrig = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2HSV)
    ho, so, vo = cv2.split(hsvOrig)

    # h = cv2.multiply(h,0.99)
    # v = cv2.multiply(v, 0.90)
    s = cv2.multiply(s, params["Smultiply"])
    s = cv2.add(s, params["Sadd"])
    h = np.clip(h, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    s = np.clip(s, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    if show_steps:
        show_histogram(h, ho, "Hue After", "Hue Original")
        show_histogram(v, vo, "Value After", "Value Original")
        show_histogram(s, so, "Saturation After", "Saturation Original")


    # BGR modifier
    b, g, r = cv2.split(frame)
    bo, go, ro = cv2.split(frameOrig)
    if show_steps:
        cv2.imwrite("output/BGR-edit_frame.jpg", frame)
        show_histogram(r, ro, "Red After", "Red Original")
        show_histogram(g, go, "Green After", "Green Original")
        show_histogram(b, bo, "Blue After", "Blue Original")

    #y = scipy.ndimage.median_filter(y, (params["YfilterSize"], params["YfilterSize"]))
    mask = getGaussian2D(r.shape, params["sigma"], show_steps)
    r = cv2.multiply(r.astype(np.float64), 1 + params["gausRAdj"] * mask)
    g = cv2.multiply(g.astype(np.float64), 1 + params["gausGAdj"] * mask)
    b = cv2.multiply(b.astype(np.float64), 1 + params["gausBAdj"] * mask)
    r = cv2.subtract(r, params["Rsubstract"])
    r = cv2.multiply(r, params["Rmultiply"])
    b = cv2.subtract(b, params["Bsubstract"])
    b = cv2.multiply(b, params["Bmultiply"])
    g = cv2.subtract(g, params["Gsubstract"])
    g = cv2.multiply(g, params["Gmultiply"])

    r = np.clip(r,0,255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    frame = cv2.merge((b,g,r))
    if show_steps:
        cv2.imwrite("output/BGR-edit_frame.jpg", frame)
        show_histogram(r, ro, "Red After", "Red Original")
        show_histogram(g, go, "Green After", "Green Original")
        show_histogram(b, bo, "Blue After", "Blue Original")

    if evaluate:
        mae = cv2.mean(np.abs(cv2.subtract(frame.astype(np.float64), frameOrig.astype(np.float64))))
        mse = cv2.mean(cv2.pow(cv2.subtract(frame.astype(np.float64), frameOrig.astype(np.float64)), 2))
        psnr = (10 * cv2.log(cv2.divide(255 ** 2, mse))).T[0]
        print(
            f"MAE: B={mae[0]:.3f} G={mae[1]:.3f} R={mae[2]:.3f}\tMSE: B={mse[0]:.3f} G={mse[1]:.3f} R={mse[2]:.3f}\tPSNR: B={psnr[0]:.3f} G={psnr[1]:.3f} R={psnr[2]:.3f}")
        return frame, list(mae[:3]), list(mse[:3]), list(psnr[:3])
    return frame, [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]

def process_video(input_path:str,original:str, output_path:str,color_params:dict, show_steps=False, evaluate=False, show_processed_frame=True,** kwargs):
    print("Processing "+input_path)
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    capOrig = cv2.VideoCapture(original)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened() and capOrig.isOpened():
        ret, frame = cap.read()
        ret, frameOrig = capOrig.read()
        if not ret:
            break

        #frameOut, mae, mse, psnr = process_frame(frame, frameOrig, show_steps, evaluate, Umultiply=2.5, Usubstract=76)
        frameOut, mae, mse, psnr = process_frame2(frame, frameOrig, show_steps, evaluate, color_params)
        out.write(frameOut)

        if show_processed_frame:
            cv2.imshow('Processing Video', frameOut)
        if evaluate or show_steps or cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

    params = {  # Standaardwaarden
        "filterSize": 5,
        "sigma": 1080 / 2,
        "gausRAdj": 0.25,
        "gausGAdj": 0,
        "gausBAdj": 0.1,
        "Rmultiply": 0.8,
        "Rsubstract": -5,
        "Gmultiply": 0.8,
        "Gsubstract": -5,
        "Bmultiply": 0.7,
        "Bsubstract": -5,
        "Sadd": 10,
        "Smultiply": 1.5
    }
    process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "../SourceVideos/2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "output/2017-01-07_President_Obama's_Weekly_Address.mp4",
                  params,evaluate=True)
    process_video("../DegradedVideos/archive_20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "../SourceVideos/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "output/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  params, evaluate=True)
    #process_video("../DegradedVideos/archive_Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
    #              "../SourceVideos/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
    #              "output/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4")
    #process_video("../DegradedVideos/archive_Robin_Singing_video.mp4",
    #              "../SourceVideos/Robin_Singing_video.mp4",
    #              "output/Robin_Singing_video.mp4")
    #process_video("../DegradedVideos/archive_Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
    #              "../SourceVideos/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
    #              "output/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4")
    #process_video("../ArchiveVideos/Apollo_11_Landing_-_first_steps_on_the_moon.mp4",
    #              "../ArchiveVideos/Apollo_11_Landing_-_first_steps_on_the_moon.mp4",
    #              "output/Apollo_11_Landing_-_first_steps_on_the_moon.mp4")
    #process_video("..\ArchiveVideos\Breakfast-at-tiffany-s-official®-trailer-hd.mp4",
    #              "..\ArchiveVideos\Breakfast-at-tiffany-s-official®-trailer-hd.mp4",
    #              "output\Breakfast-at-tiffany-s-official®-trailer-hd.mp4")
    #process_video("..\ArchiveVideos\Edison_speech,_1920s.mp4",
    #              "..\ArchiveVideos\Edison_speech,_1920s.mp4",
    #              "output\ArchiveVideos\Edison_speech,_1920s.mp4")
    #process_video("..\ArchiveVideos\President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962.mp4",
    #              "..\ArchiveVideos\President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962.mp4",
    #              "output\ArchiveVideos\President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962.mp4")
    #process_video("..\ArchiveVideos\The_Dream_of_Kings.mp4",
    #              "..\ArchiveVideos\The_Dream_of_Kings.mp4",
    #              "output\The_Dream_of_Kings.mp4")

if __name__ == '__main__':
    main()