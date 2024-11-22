import math
import random
import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.io import wavfile
import moviepy

def cumulative_histogram(image):
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    cumhist = np.cumsum(hist).astype(np.float64)
    cumhist = cumhist/cumhist[-1]
    return cumhist

def show_results(image,image2, nameImg="",nameImg2=""):
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(321)
    ax.set_title(nameImg)
    ax0 = fig.add_subplot(322)
    ax0.set_title(nameImg2)
    ax1 = fig.add_subplot(323)
    ax1.set_title("Histogram of "+nameImg)
    ax2 = fig.add_subplot(324, sharex=ax1)
    ax2.set_title("Histogram of "+nameImg2)
    ax3 = fig.add_subplot(325, sharex=ax1)
    ax3.set_title("Cumulative Histogram of "+nameImg)
    ax4 = fig.add_subplot(326, sharex=ax1)
    ax4.set_title("Cumulative Histogram of "+nameImg2)
    ax1.set_xlim([0,255])
    ax.imshow(image, cmap=plt.get_cmap("gray"))
    ax0.imshow(image2, cmap=plt.get_cmap("gray"))
    ax1.plot(cv2.calcHist([image],[0],None,[256],[0,256]))
    ax2.plot(cv2.calcHist([image2],[0],None,[256],[0,256]))
    ax3.plot(cumulative_histogram(image))
    ax4.plot(cumulative_histogram(image2))
    plt.show()

def histogram_matching(inframe,orframe,show=False,type=""):
    if show:
        show_results(orframe, inframe,type+" original",type+" before")
    ref = cumulative_histogram(orframe)
    low = cumulative_histogram(inframe)
    low_normalized = low / low.max()
    ref_normalized = ref / ref.max()

    lookup_table = np.zeros(256, dtype=np.uint8)
    j = 1
    for i in range(256):
        while j < 256 and ref_normalized[j] <= low_normalized[i]:
            j += 1
        lookup_table[i] = j - 1

    outframe = cv2.LUT(inframe, lookup_table)

    if show:
        show_results(orframe, outframe,type+" original",type+" matched")
    return outframe

def process_frame(frame, frameOrig,show_steps=False):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsvOrig = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2HSV)
    ho, so, vo = cv2.split(hsvOrig)

    if show_steps:
        show_results(h, ho, "Hue", "Hue Original")
        show_results(v, vo, "Value", "Value Original")
        show_results(s, so, "Saturation", "Saturation Original")

    #h = scipy.ndimage.maximum_filter(h,(5,5))
    #l = scipy.ndimage.maximum_filter(l, (3,3))
    #s = scipy.ndimage.maximum_filter(s, (5,5))

    #histogram_matching(h,ho,show_steps,"Hue")
    #histogram_matching(v, vo, show_steps, "Value")
    #histogram_matching(s, so,show_steps,"Saturation")

    #h = cv2.add(h,15)
    #l = cv2.add(l, 80)
    s = cv2.add(s, 60)

    #s = scipy.ndimage.gaussian_filter(s,3)

    #h = cv2.normalize(h, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    v = cv2.normalize(v,None,0,255,norm_type=cv2.NORM_MINMAX)
    #s = cv2.normalize(s,None,0,255,norm_type=cv2.NORM_MINMAX)
    #l = np.clip(l, 0, 255)
    s = np.clip(s, 0, 255)
    h = np.clip(h, 0, 255)

    if show_steps:
        show_results(h, ho, "Hue", "Hue Original")
        show_results(v, vo, "Value", "Value Original")
        show_results(s, so, "Saturation", "Saturation Original")

    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # output_0 = cv2.normalize(frame[:, :, 0],None,0,255,norm_type=cv2.NORM_MINMAX)
    # output_1 = cv2.normalize(frame[:, :, 1], None, 0, 255, norm_type=cv2.NORM_MINMAX)
    # output_2 = cv2.normalize(frame[:, :, 2], None, 0, 255, norm_type=cv2.NORM_MINMAX)
    # output_0 = np.clip(output_0, 0, 255)
    # output_1 = np.clip(output_1, 0, 255)
    # output_2 = np.clip(output_2, 0, 255)

    # Merge the color channels back together
    # frame = cv2.merge((frame[:, :, 0], frame[:, :, 1], frame[:, :, 2])).astype(np.uint8)

    return frame

def process_video(input_path,original, output_path, show_steps=False, show_processed_frame=True):
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

        frame = process_frame(frame,frameOrig,show_steps)
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

    process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "../SourceVideos/2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "output/2017-01-07_President_Obama's_Weekly_Address.mp4")
    process_video("../DegradedVideos/archive_20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "../SourceVideos/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "output/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4")
    process_video("../DegradedVideos/archive_Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
                  "../SourceVideos/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
                  "output/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4")
    process_video("../DegradedVideos/archive_Robin_Singing_video.mp4",
                  "../SourceVideos/Robin_Singing_video.mp4",
                  "output/Robin_Singing_video.mp4")
    process_video("../DegradedVideos/archive_Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
                  "../SourceVideos/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
                  "output/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4")

if __name__ == '__main__':
    main()