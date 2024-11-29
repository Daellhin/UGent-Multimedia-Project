# Print iterations progress
def printProgressBar(iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def mono_to_stereo(left_chanel: list[float], right_chanel: list[float]):
    return [[left_chanel[i], right_chanel[i]] for i in range(len(left_chanel))]


def stereo_to_mono(stereo_audio: list[list[float]]):
    left_channel = [sample[0] for sample in stereo_audio]
    right_channel = [sample[1] for sample in stereo_audio]
    return (left_channel, right_channel)

def stereo_calculate_MSE(
    original: list[list[float]], filtered: list[list[float]], n_samples=200000
):
    """
    Returns Mean Squared quadratic Error(MSE) for both chanels for the n first samples
    """
    left_original, right_original = stereo_to_mono(original)
    left_filtered, right_filtered = stereo_to_mono(filtered)
    mse_left = (
        sum((left_filtered[i] - left_original[i]) ** 2 for i in range(n_samples))
        / n_samples
    )
    mse_right = (
        sum((right_filtered[i] - right_original[i]) ** 2 for i in range(n_samples))
        / n_samples
    )
    return (mse_left, mse_right)
