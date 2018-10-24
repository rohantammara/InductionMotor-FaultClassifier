import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
"""
(MEET NOTES)
- Use Power Spectrum to differentiate between Healthy and Faulty base frequency (50 Hz)
- 0.9853
- Classify at a single loading also
- Use only 0, 25, 50 % loading (first 50%)
"""
# Number of levels (of decomposition)
L = 6

def wavelet_transform(wave, wavelet, level):
    # Wavelet Transform
    wavelet = pywt.Wavelet(wavelet)
    coeffs = pywt.wavedec(wave, wavelet=wavelet, level=level)
    details_reconstructed = np.zeros(len(wave))
    for i in range(level):
        details_reconstructed = np.vstack((details_reconstructed,
         pywt.upcoef(part='d', coeffs=coeffs[i+1], wavelet=wavelet, level=level-(i), take=len(wave))
         ))

    return details_reconstructed[1:]

def prepare_data():
    # Load data
    M = 720000
    N = 900000

    print('Fetching data')
    df = pd.concat([pd.read_csv('data/i1.csv'), pd.read_csv('data/i2.csv'), pd.read_csv('data/i3.csv'), pd.read_csv('data/y.csv')], axis=1)

    df_train = pd.concat([df[:M], df[N:N+M], df[2*N:2*N+M], df[3*N:3*N+M]])
    df_test = pd.concat([df[M:N], df[N+M:2*N], df[2*N+M:3*N], df[3*N+M:4*N]])

    x_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, 3].values
    x_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, 3].values
    print('done')

    del [df, df_train, df_test]

    # Preprocessing

    print('Performing wavelet transforms')
    i1_train = wavelet_transform(x_train[:, 0], wavelet='db4', level=L)
    i2_train = wavelet_transform(x_train[:, 1], wavelet='db4', level=L)
    i3_train = wavelet_transform(x_train[:, 2], wavelet='db4', level=L)
    i1_test = wavelet_transform(x_test[:, 0], wavelet='db4', level=L)
    i2_test = wavelet_transform(x_test[:, 1], wavelet='db4', level=L)
    i3_test = wavelet_transform(x_test[:, 2], wavelet='db4', level=L)
    print('done')

    del [x_train, x_test]

    print('Preparing training data')
    x_train = np.concatenate((i1_train, i2_train, i3_train), axis=0)
    x_train = np.transpose(x_train)
    y_train = to_categorical(y_train)
    print('done')

    print('Preparing test data')
    x_test = np.concatenate((i1_test, i2_test, i3_test), axis=0)
    x_test = np.transpose(x_test)
    y_test = to_categorical(y_test)
    print('done')

    del [i1_train, i1_test, i2_train, i2_test, i3_train, i3_test]

    return x_train, y_train, x_test, y_test

def network(input_shape):
    # Network Arch.
    net = Sequential()
    net.add(Dense(64, input_shape=input_shape, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dropout(0.2))
    net.add(Dense(128, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dense(128, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dropout(0.4))
    net.add(Dense(256, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dense(4, kernel_initializer='lecun_uniform', activation='softmax'))

    optimizer = Adam(lr=0.001)

    net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return net

def train(x_train, y_train, x_test, y_test):
    # Training
    NUM_EPOCHS = 50
    BATCH_SIZE = 1024

    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.1, patience=3, min_lr=0.0001)
    savepoint = ModelCheckpoint(filepath="wavelet_model.hdf5", monitor='val_acc', save_best_only=True, mode='max')

    clf = network(input_shape=(L*3,))

    clf.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
            shuffle=True, callbacks=[savepoint, reduce_lr], validation_data=(x_test, y_test))

if __name__ == "__main__":

    X, Y, x, y = prepare_data()
    print('Starting training')
    train(X, Y, x, y)
    print('finished')
