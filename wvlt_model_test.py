import time
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, load_model

### Constants ###
# Number of levels of wavelet decomposition
L = 4
# Width of window
W = 250
# Validation batch size
BATCH_SIZE = 32

def wavelet_transform(wave, wavelet, level):
    """
    Performs wavelet transform
    """
    wavelet = pywt.Wavelet(wavelet)
    coeffs = pywt.wavedec(wave, wavelet=wavelet, level=level)
    return np.concatenate(coeffs[1:])

def prepare_data():
    """
    Prepare training and test data
    """

    df_3 = pd.concat([pd.read_csv('test_data/i1_3.csv'), pd.read_csv('test_data/i2_3.csv'), pd.read_csv('test_data/i3_3.csv')], axis=1)
    df_h = pd.concat([pd.read_csv('test_data/i1_h.csv'), pd.read_csv('test_data/i2_h.csv'), pd.read_csv('test_data/i3_h.csv')], axis=1)

    x_test = pd.concat([df_3, df_h], axis=0).values
    #noise = 2*np.random.normal(0, 0.01, (200000, 3))
    #x_test = x_test + noise

    y_test = to_categorical(np.repeat([0, 3], 400))

    i1_test = []
    i2_test = []
    i3_test = []

    wvlt = 'db6'
    for i in range(800):
        i1_test.append(wavelet_transform(x_test[W*i:W*(i+1), 0], wavelet=wvlt, level=L))
        i2_test.append(wavelet_transform(x_test[W*i:W*(i+1), 1], wavelet=wvlt, level=L))
        i3_test.append(wavelet_transform(x_test[W*i:W*(i+1), 2], wavelet=wvlt, level=L))

    x_test = np.hstack((np.array(i1_test), np.array(i2_test), np.array(i3_test)))

    return x_test, y_test

if __name__ == "__main__":

    x, y = prepare_data()

    clf = load_model("wavelet_model.hdf5")
    scores = clf.evaluate(x, y, 32)
    print("Loss: ", scores[0])
    print("Accuracy: ", scores[1])
