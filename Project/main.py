import math
import random
import cv2
import numpy as np
import scipy
from PIL.ImageChops import multiply
from matplotlib import pyplot as plt
from scipy.io import wavfile
import moviepy
from visualisations import *

def process_frame(frame, frameOrig, show_steps=False):
    # YUV modifier - kringverzwakking
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    yuv_or = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YUV)
    yo, uo, vo = cv2.split(yuv_or)

    if show_steps:
        cv2.imwrite("output/start_Frame.jpg", frame)
        # show_histogram(y, yo, "Y", "Y Original")
        # show_spectrum(y, yo, "Y")
        # show_histogram(u, uo, "U", "U original")
        # show_spectrum(u, uo, "U")
        # show_histogram(v, vo, "V", "V Original")
        # show_spectrum(v, vo, "V")

    # y = scipy.ndimage.median_filter(y, (3,3))
    y = cv2.blur(y, (5, 5))

    rows, cols = v.shape
    kernel_x = cv2.getGaussianKernel(cols, 1080 / 2)
    kernel_y = cv2.getGaussianKernel(rows, 1080 / 2)
    kernel = kernel_y * kernel_x.T
    mask = 1 - kernel / np.linalg.norm(kernel)
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
    if show_steps:
        plt.imshow(mask)
        plt.show()
    y = cv2.multiply(y.astype(np.float64), 3 / 4 + 1 / 2 * mask)
    u = cv2.multiply(u.astype(np.float64), 1 + 0.01 * mask)
    v = cv2.multiply(v.astype(np.float64), 1 + 0.01 * mask)
    u = cv2.multiply(u, 2.5)
    u = cv2.subtract(u, 190)
    v = cv2.multiply(v, 2.1)
    v = cv2.subtract(v, 140)

    # u = scipy.ndimage.gaussian_filter(u,1.2)
    # v = scipy.ndimage.gaussian_filter(v, 1.2)
    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    frame = cv2.merge((y, u, v))

    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    if show_steps:
        cv2.imwrite("output/YUV-edit_Frame.jpg", frame)
        show_histogram(y, yo, "Y edit", "Y Original")
        # show_spectrum(y, yo, "Y")
        show_histogram(u, uo, "U edit", "U original")
        # show_spectrum(u, uo, "U")
        show_histogram(v, vo, "V edit", "V Original")
        # show_spectrum(v, vo, "V")

    # BGR modifier
    b, g, r = cv2.split(frame)
    bo, go, ro = cv2.split(frameOrig)

    if show_steps:
        show_histogram(r, ro, "Red", "Red Original")
        show_spectrum(r, ro, "Red")
        show_histogram(g, go, "Green", "Green Original")
        show_spectrum(g, go, "Green")
        show_histogram(b, bo, "Blue", "Blue Original")
        show_spectrum(b, bo, "Blue")

    # r = scipy.ndimage.gaussian_filter(r, 2)
    # g = scipy.ndimage.gaussian_filter(g, 2.5)
    # b = scipy.ndimage.gaussian_filter(b, 2)
    # r = cv2.multiply(r,0.80)
    # g = cv2.multiply(g,0.70)
    # b = cv2.multiply(b,0.6)

    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    frame = cv2.merge((b, g, r))

    # HSV modifiers
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsvOrig = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2HSV)
    ho, so, vo = cv2.split(hsvOrig)

    if show_steps:
        cv2.imwrite("output/BGR-edit_frame.jpg", frame)
        # show_histogram(h, ho, "Hue", "Hue Original")
        # show_spectrum(h,ho,"Hue")
        # show_histogram(v, vo, "Value", "Value Original")
        # show_spectrum(v,vo,"Value")
        # show_histogram(s, so, "Saturation", "Saturation Original")
        # show_spectrum(s,so,"Saturation")

    # h = scipy.ndimage.median_filter(h,(1,3))
    # s = scipy.ndimage.median_filter(s, (5,5))
    # v = scipy.ndimage.median_filter(v,(3,3))

    # h = cv2.multiply(h,0.99)
    # v = cv2.multiply(v, 0.90)
    # s = cv2.multiply(s,1.75)
    s = cv2.add(s, 20)

    # s = scipy.ndimage.gaussian_filter(s, 4)
    h = np.clip(h, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    s = np.clip(s, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    if show_steps:
        show_histogram(h, ho, "Hue", "Hue Original")
        show_histogram(v, vo, "Value", "Value Original")
        show_histogram(s, so, "Saturation", "Saturation Original")

    return frame


def process_video(input_path, original, output_path, show_steps=False, show_processed_frame=True):
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

        frame = process_frame(frame, frameOrig, show_steps)
        # Write the processed frame
        out.write(frame)

        if show_processed_frame:
            cv2.imshow('Processing Video', frame)
        if show_steps or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "../SourceVideos/2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "output/2017-01-07_President_Obama's_Weekly_Address.mp4")
    process_video("../DegradedVideos/archive_20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "../SourceVideos/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "output/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",True)
    # process_video("../DegradedVideos/archive_Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
    #              "../SourceVideos/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
    #              "output/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4")
    # process_video("../DegradedVideos/archive_Robin_Singing_video.mp4",
    #              "../SourceVideos/Robin_Singing_video.mp4",
    #              "output/Robin_Singing_video.mp4")
    # process_video("../DegradedVideos/archive_Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
    #              "../SourceVideos/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
    #              "output/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4")


if __name__ == '__main__':
    main()