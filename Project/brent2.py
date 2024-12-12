from dataclasses import dataclass
import skimage
from visualisations import *

@dataclass
class ColorParams():
    #eerste filter voor bewerking
    FilterSize :int = 3
    #kleuraanpassing met gausiaanse vorm (hoog aan randen)
    GaussianSize :int = 1080
    UGaussianAdjust :float = 0
    VGaussianAdjust :float = 0
    YGaussianAdjust :float = 0
    #YUV-kleurcorrecties
    UMultiply :float = 1
    VMultiply :float = 1
    YSubstract :int = 0
    USubstract :int = 0
    VSubstract :int = 0
    #HSV-kleurcorrecties
    SaturationAdd :int = 0
    SaturationMultiply :float = 1
    ValueAdd :int = 0
    ValueMultiply :float = 1


@dataclass
class Enablers():
    rek :bool = False
    show_color_steps :bool = False
    show_processed_frame :bool = False
    evaluate :bool = False

def color_adjust(frame:cv2.typing.MatLike, frameOrig:cv2.typing.MatLike,params:ColorParams,enable:Enablers ,show_steps=False,evaluate=False) -> tuple[cv2.typing.MatLike, float, float, float]:
    frame = cv2.blur(frame,(params.FilterSize,params.FilterSize))

    # YUV modifier - kringverzwakking
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    if show_steps:
        cv2.imwrite("output/start_Frame.jpg", frame)
        yuv_or = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YUV)
        yo, uo, vo = cv2.split(yuv_or)
        show_histogram(y, yo, "Y", "Y Original")
        show_histogram(u, uo, "U", "U original")
        show_histogram(v, vo, "V", "V Original")

    # y = scipy.ndimage.median_filter(y, (3,3))
    rows, cols = v.shape
    kernel_x = cv2.getGaussianKernel(cols, params.GaussianSize)
    kernel_y = cv2.getGaussianKernel(rows, params.GaussianSize)
    kernel = kernel_y * kernel_x.T
    mask = 1 - kernel / np.linalg.norm(kernel)
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
    if show_steps:
        plt.imshow(mask)
        plt.show()
    y = cv2.multiply(y.astype(np.float64), 1 - params.YGaussianAdjust / 2 + params.YGaussianAdjust * mask)
    u = cv2.multiply(u.astype(np.float64), 1 + params.UGaussianAdjust * mask)
    v = cv2.multiply(v.astype(np.float64), 1 + params.VGaussianAdjust * mask)
    y = cv2.subtract(y, params.YSubstract)
    u = cv2.multiply(u, params.UMultiply)
    u = cv2.subtract(u, params.USubstract)
    v = cv2.multiply(v, params.VMultiply)
    v = cv2.subtract(v, params.VSubstract)
    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    if show_steps:
        yuv_or = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YUV)
        yo, uo, vo = cv2.split(yuv_or)
        show_histogram(y, yo, "Y edit", "Y Original")
        show_histogram(u, uo, "U edit", "U original")
        show_histogram(v, vo, "V edit", "V Original")

    frame = cv2.merge((y, u, v))
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    # BGR analyser
    if show_steps:
        cv2.imwrite("output/YUV-edit_Frame.jpg", frame)
        b, g, r = cv2.split(frame)
        bo, go, ro = cv2.split(frameOrig)
        show_histogram(r, ro, "Red", "Red Original")
        show_spectrum(r, ro, "Red")
        show_histogram(g, go, "Green", "Green Original")
        show_spectrum(g, go, "Green")
        show_histogram(b, bo, "Blue", "Blue Original")
        show_spectrum(b, bo, "Blue")

    # HSV modifiers
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if show_steps:
        cv2.imwrite("output/BGR-edit_frame.jpg", frame)
        hsvOrig = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2HSV)
        ho, so, vo = cv2.split(hsvOrig)
        show_histogram(h, ho, "Hue", "Hue Original")
        show_histogram(v, vo, "Value", "Value Original")
        show_histogram(s, so, "Saturation", "Saturation Original")

    v = cv2.multiply(v, params.ValueMultiply)
    v = cv2.add(v, params.ValueAdd)
    s = cv2.multiply(s,params.SaturationMultiply)
    s = cv2.add(s, params.SaturationAdd)

    h = np.clip(h, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    s = np.clip(s, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    if show_steps:
        cv2.imwrite("output/eind_frame.jpg", frame)
        hsvOrig = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2HSV)
        ho, so, vo = cv2.split(hsvOrig)
        show_histogram(h, ho, "Hue", "Hue Original")
        show_histogram(v, vo, "Value", "Value Original")
        show_histogram(s, so, "Saturation", "Saturation Original")

    if evaluate:
        mse = skimage.metrics.mean_squared_error(frameOrig, frame)  # naar 0!
        psnr = skimage.metrics.peak_signal_noise_ratio(frameOrig, frame)
        ssim = skimage.metrics.structural_similarity(frameOrig, frame, channel_axis=-1)  # naar 1!
        # print(mse,psnr,ssim)
        return frame, mse, psnr, ssim
    return frame, -1, -1, -1