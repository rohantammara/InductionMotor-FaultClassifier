import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, SGD
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
    O = int(M/4)
    N = 900000

    print('Fetching data')
    df = pd.concat([pd.read_csv('data/i1.csv'), pd.read_csv('data/i2.csv'), pd.read_csv('data/i3.csv'), pd.read_csv('data/y.csv')], axis=1)

    df_train = pd.concat([df[:M], df[N:N+M], df[2*N:2*N+M], df[3*N:3*N+M]])
    df_test = pd.concat([df[M:N], df[N+M:2*N], df[2*N+M:3*N], df[3*N+M:4*N]])

    x_train = df_train.iloc[:, :-1].values
    x_test = df_test.iloc[:, :-1].values
    y_train = to_categorical(df_train['y'].values)
    y_test = to_categorical(df_test['y'].values)
    #y_train = to_categorical(np.repeat([[0, 1, 2, 3]], M, axis=0).flatten())
    #y_test = to_categorical(np.repeat([[0, 1, 2, 3]], O, axis=0).flatten())
    print('done')

    del [df, df_train, df_test]

    # Preprocessing
    '''
    print('Performing wavelet transforms')
    wvlt = 'db13'
    i1_train = [wavelet_transform(x_train[:M, 0], wavelet=wvlt, level=L),
                wavelet_transform(x_train[M:2*M, 0], wavelet=wvlt, level=L),
                wavelet_transform(x_train[2*M:3*M, 0], wavelet=wvlt, level=L),
                wavelet_transform(x_train[3*M:, 0], wavelet=wvlt, level=L)]

    i2_train = [wavelet_transform(x_train[:M, 1], wavelet=wvlt, level=L),
                wavelet_transform(x_train[M:2*M, 1], wavelet=wvlt, level=L),
                wavelet_transform(x_train[2*M:3*M, 1], wavelet=wvlt, level=L),
                wavelet_transform(x_train[3*M:, 1], wavelet=wvlt, level=L)]

    i3_train = [wavelet_transform(x_train[:M, 2], wavelet=wvlt, level=L),
                wavelet_transform(x_train[M:2*M, 2], wavelet=wvlt, level=L),
                wavelet_transform(x_train[2*M:3*M, 2], wavelet=wvlt, level=L),
                wavelet_transform(x_train[3*M:, 2], wavelet=wvlt, level=L)]

    i1_test = [wavelet_transform(x_test[:O, 0], wavelet=wvlt, level=L),
                wavelet_transform(x_test[O:2*O, 0], wavelet=wvlt, level=L),
                wavelet_transform(x_test[2*O:3*O, 0], wavelet=wvlt, level=L),
                wavelet_transform(x_test[3*O:, 0], wavelet=wvlt, level=L)]

    i2_test = [wavelet_transform(x_test[:O, 1], wavelet=wvlt, level=L),
                wavelet_transform(x_test[O:2*O, 1], wavelet=wvlt, level=L),
                wavelet_transform(x_test[2*O:3*O, 1], wavelet=wvlt, level=L),
                wavelet_transform(x_test[3*O:, 1], wavelet=wvlt, level=L)]

    i3_test = [wavelet_transform(x_test[:O, 2], wavelet=wvlt, level=L),
                wavelet_transform(x_test[O:2*O, 2], wavelet=wvlt, level=L),
                wavelet_transform(x_test[2*O:3*O, 2], wavelet=wvlt, level=L),
                wavelet_transform(x_test[3*O:, 2], wavelet=wvlt, level=L)]
    print('done')

    """ # Plots
    plt.subplot(4,1,1)
    plt.plot(x_train[2160000:2165000, 0])
    plt.subplot(4,1,2)
    plt.plot(i1_train[1])
    plt.subplot(4,1,3)
    plt.plot(i2_train[1])
    plt.subplot(4,1,4)
    plt.plot(i3_train[1])
    plt.show()
    """

    del [x_train, x_test]

    print('Preparing training data')
    x_train = np.concatenate((np.concatenate((i1_train[0], i2_train[0], i3_train[0]), axis=0),
                            np.concatenate((i1_train[1], i2_train[1], i3_train[1]), axis=0),
                            np.concatenate((i1_train[2], i2_train[2], i3_train[2]), axis=0),
                            np.concatenate((i1_train[3], i2_train[3], i3_train[3]), axis=0)), axis=1)

    x_train = np.transpose(x_train)
    print('done')

    print('Preparing test data')
    x_test = np.concatenate((np.concatenate((i1_test[0], i2_test[0], i3_test[0]), axis=0),
                            np.concatenate((i1_test[1], i2_test[1], i3_test[1]), axis=0),
                            np.concatenate((i1_test[2], i2_test[2], i3_test[2]), axis=0),
                            np.concatenate((i1_test[3], i2_test[3], i3_test[3]), axis=0)), axis=1)

    x_test = np.transpose(x_test)
    print('done')

    del [i1_test, i2_test, i3_test, i1_train, i2_train, i3_train]
    '''
    return x_train, y_train, x_test, y_test

def network(input_shape):
    # Network Arch. (Use RNN maybe)
    net = Sequential()
    net.add(Dense(128, input_shape=input_shape, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dense(256, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dropout(0.2))
    net.add(Dense(256, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dense(512, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dropout(0.4))
    net.add(Dense(512, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dense(4, kernel_initializer='lecun_uniform', activation='softmax'))

    optimizer = Adam(lr=0.001)

    net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return net

def train(x_train, y_train, x_test, y_test):
    # Training
    NUM_EPOCHS = 10
    BATCH_SIZE = 1024

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, min_lr=0.0001)
    savepoint = ModelCheckpoint(filepath="wavelet_model.hdf5", monitor='val_acc', save_best_only=True, mode='max')

    clf = network(input_shape=(3,))

    clf.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
            shuffle=True, callbacks=[savepoint, reduce_lr], validation_data=(x_test, y_test))

if __name__ == "__main__":

    X, Y, x, y = prepare_data()
    print('Starting training')
    train(X, Y, x, y)

    del [X, Y, x, y]
    print('finished')
