import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
"""
(MEET NOTES)
- Use Power Spectrum to differentiate between Healthy and Faulty base frequency (50 Hz)
- 0.9853
- Classify at a single loading also
- Use only 0, 25, 50 % loading (first 50%)
- 
"""
def wavelet_transform(wave):
    wavelet = pywt.Wavelet('haar')
    max_level = pywt.dwt_max_level(len(wave), wavelet)
    coeffs = pywt.wavedec(wave, wavelet=wavelet, level=max_level)
    detail_reconstruct = []
    for i in range(max_level):
        detail_reconstruct.append(
        pywt.upcoef(part='d', coeffs=coeffs[i+1], wavelet=wavelet, level=max_level-(i))
                                )
    return detail_reconstruct

def base_classifier(input_shape):
    net = Sequential()
    net.add(Dense(128, input_shape=input_shape, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Flatten())
    net.add(Dense(256, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dense(256, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dropout(0.4))
    net.add(Dense(512, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dense(1, kernel_initializer='lecun_uniform', activation='sigmoid'))

    optimizer = Adam(lr=0.001)

    net.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return net

def main():
    df = pd.concat([pd.read_csv('diff_i1_2.csv'), pd.read_csv('i3.csv'), pd.read_csv('y.csv')], axis=1)

    y_data = df['y'].values
    x_data = df['i3'].values

    # Constants
    M = 1259776
    N = 539904
    P = 4921
    Q = 2109
    R = 256

    x_train = x_data[:M]
    y_train = y_data[:M]
    x_test = x_data[M:M+N]
    y_test = y_data[M:M+N]

    x_train = np.reshape(x_train, (P, R))
    x_test = np.reshape(x_test, (Q, R))
    y_train = np.reshape(y_train, (P, R))
    y_test = np.reshape(y_test, (Q, R))

    Y_train = []
    Y_test = []

    for _ in range(P):
        if sum(y_train[_]) != 0:
            Y_train.append(1)
        else:
            Y_train.append(0)

    for _ in range(Q):
        if sum(y_test[_]) != 0:
            Y_test.append(1)
        else:
            Y_test.append(0)

    X_train = np.empty([P, 8, R])
    X_test = np.empty([Q, 8, R])

    for _ in range(P):
        X_train[_] = wavelet_transform(x_train[_])

    for _ in range(Q):
        X_test[_] = wavelet_transform(x_train[_])

    NUM_EPOCHS = 50
    BATCH_SIZE = 32

    savepoint = ModelCheckpoint(filepath="main_model.hdf5", monitor='val_acc', save_best_only=True, mode='max')

    clf = base_classifier(input_shape=(8, R))

    clf.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, callbacks=[savepoint], validation_data=(X_test, Y_test))

if __name___ == "__main__"
    # Start main
    main()
