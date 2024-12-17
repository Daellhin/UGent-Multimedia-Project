import numpy
import cv2
import matplotlib
import scipy
import sys
import audioplayer
import skimage
import tqdm
import moviepy

def main():
    print("Versies:")
    print("\tPython:\t" + sys.version.split(" ")[0])
    print("\tOpencv:\t" + cv2.__version__)
    print("\tNumpy:\t" + numpy.__version__)
    print("\tScipy:\t" + scipy.__version__)
    print("\tMatplotlib:\t" + matplotlib.__version__)
    print("\tAudioplayer:\t" + audioplayer.__version__)
    print("\tSkimage:\t" + skimage.__version__)
    print("\tTqdm:\t" + tqdm.__version__)
    print("\tMoviepy:\t" + moviepy.__version__)

# main code here...
if __name__ == '__main__':
    main()
