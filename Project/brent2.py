from dataclasses import dataclass
import skimage
from visualisations import *

@dataclass
class ColorParams:
    FilterSize = 5
    GaussianSize = 1080
    UGaussianAdjust = 0.005
    VGaussianAdjust = 0.005
    YGaussianAdjust = 1/3
    SaturationAdd = 20
    SaturationMultiply = 1
    UMultiply = 2.5
    VMultiply = 2.1
    USubstract = 190
    VSubstract = 140

def process_frame(frame:cv2.typing.MatLike, frameOrig:cv2.typing.MatLike,params:ColorParams ,show_steps=False,evaluate=False) -> tuple[cv2.typing.MatLike, float, float, float]:
    frame = cv2.blur(frame,(params.FilterSize,params.FilterSize))

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
    u = cv2.multiply(u, params.UMultiply)
    u = cv2.subtract(u, params.USubstract)
    v = cv2.multiply(v, params.VMultiply)
    v = cv2.subtract(v, params.VSubstract)

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
    s = cv2.multiply(s,params.SaturationMultiply)
    s = cv2.add(s, params.SaturationAdd)

    h = np.clip(h, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    s = np.clip(s, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    if show_steps:
        show_histogram(h, ho, "Hue", "Hue Original")
        show_histogram(v, vo, "Value", "Value Original")
        show_histogram(s, so, "Saturation", "Saturation Original")

    if evaluate:
        cv2.imwrite("output/evaluate_frame.jpg", frame)
        mse = skimage.metrics.mean_squared_error(frameOrig, frame)  # naar 0!
        psnr = skimage.metrics.peak_signal_noise_ratio(frameOrig, frame)
        ssim = skimage.metrics.structural_similarity(frameOrig, frame, channel_axis=-1)  # naar 1!
        # print(mse,psnr,ssim)
        return frame, mse, psnr, ssim
    return frame, -1, -1, -1