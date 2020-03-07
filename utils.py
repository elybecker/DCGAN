import numpy as np

def getNextBatch(x, batch_size):
    nImg = x.shape[0]
    idx = np.random.randint(0, nImg, size=batch_size)
    return x[idx, :]


def normalizeImages(x):
    # normalise to -1 to 1, tanH
    v_min = x.min(axis=(1, 2), keepdims=True)
    v_max = x.max(axis=(1, 2), keepdims=True)
    return 2. * (x - v_min)/(v_max - v_min) - 1.
    

def reScale(x):
    return (x + 1)/2
