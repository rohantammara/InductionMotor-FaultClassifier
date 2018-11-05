import time
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical

### Constants ###
# Number of folds for k-fold cross validation
folds = 5
# Number of levels of wavelet decomposition
L = 4
# Width of window
W = 250
# Auxiliaries
N = 900000/folds
P = int(2880000/(W*4))
R = int(720000/(W*4))
# Number of epochs
NUM_EPOCHS = 15
# Batch size
BATCH_SIZE = 32

def wavelet_transform(wave, wavelet, level):
    """
    Performs wavelet transform
    """
    wavelet = pywt.Wavelet(wavelet)
    coeffs = pywt.wavedec(wave, wavelet=wavelet, level=level)
    return np.concatenate(coeffs[1:])

def prepare_data(k=5):
    """
    Prepare training and test data
    """
    df_1 = pd.concat([pd.read_csv('data/i1_1.csv'), pd.read_csv('data/i2_1.csv'), pd.read_csv('data/i3_1.csv')], axis=1)
    df_2 = pd.concat([pd.read_csv('data/i1_2.csv'), pd.read_csv('data/i2_2.csv'), pd.read_csv('data/i3_2.csv')], axis=1)
    df_3 = pd.concat([pd.read_csv('data/i1_3.csv'), pd.read_csv('data/i2_3.csv'), pd.read_csv('data/i3_3.csv')], axis=1)
    df_h = pd.concat([pd.read_csv('data/i1_h.csv'), pd.read_csv('data/i2_h.csv'), pd.read_csv('data/i3_h.csv')], axis=1)

    df_train = pd.concat([df_1.drop(df_1.index[int((k-1)*N):int(k*N)], axis=0),
                          df_2.drop(df_2.index[int((k-1)*N):int(k*N)], axis=0),
                          df_3.drop(df_3.index[int((k-1)*N):int(k*N)], axis=0),
                          df_h.drop(df_h.index[int((k-1)*N):int(k*N)], axis=0)], axis=0)
    df_test = pd.concat([df_1[int((k-1)*N):int(k*N)], df_2[int((k-1)*N):int(k*N)], df_3[int((k-1)*N):int(k*N)], df_h[int((k-1)*N):int(k*N)]], axis=0)

    del [df_1, df_2, df_3, df_h]

    x_train = df_train.values
    x_test = df_test.values
    y_train = to_categorical(np.repeat([0, 1, 2, 3], P))
    y_test = to_categorical(np.repeat([0, 1, 2, 3], R))

    del [df_train, df_test]

    i1_train = []
    i2_train = []
    i3_train = []
    i1_test = []
    i2_test = []
    i3_test = []

    wvlt = 'db6'
    for i in range(4*P):
        i1_train.append(wavelet_transform(x_train[W*i:W*(i+1), 0], wavelet=wvlt, level=L))
        i2_train.append(wavelet_transform(x_train[W*i:W*(i+1), 1], wavelet=wvlt, level=L))
        i3_train.append(wavelet_transform(x_train[W*i:W*(i+1), 2], wavelet=wvlt, level=L))

    for i in range(4*R):
        i1_test.append(wavelet_transform(x_test[W*i:W*(i+1), 0], wavelet=wvlt, level=L))
        i2_test.append(wavelet_transform(x_test[W*i:W*(i+1), 1], wavelet=wvlt, level=L))
        i3_test.append(wavelet_transform(x_test[W*i:W*(i+1), 2], wavelet=wvlt, level=L))

    del [x_train, x_test]

    x_train = np.hstack((np.array(i1_train), np.array(i2_train), np.array(i3_train)))
    x_test = np.hstack((np.array(i1_test), np.array(i2_test), np.array(i3_test)))

    return x_train, y_train, x_test, y_test

def network(input_shape):
    """
    Network Architecture
    """
    net = Sequential()
    net.add(Dense(128, input_shape=input_shape, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dense(256, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dropout(0.2))
    net.add(Dense(512, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dropout(0.4))
    net.add(Dense(512, kernel_initializer='lecun_uniform', activation='relu'))
    net.add(Dense(4, activation='softmax'))

    optimizer = Adam(lr=0.001)

    net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return net

def train(x_train, y_train, x_test, y_test, fold):
    """
    Helper function to train model
    """
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, min_lr=0.0001)
    savepoint = ModelCheckpoint(filepath="wavelet_model.hdf5", monitor='val_acc', save_best_only=True, mode='max')

    clf = network(input_shape=(len(x_train[0]),))

    if fold != 1:
        clf.load_weights("wavelet_model_weights.hdf5")

    clf.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
            shuffle=True, callbacks=[savepoint, reduce_lr], validation_data=(x_test, y_test))

    clf.save_weights("wavelet_model_weights.hdf5")

if __name__ == "__main__":
    # Start time
    t0 = time.time()

    # Train
    for i in range(folds):
        print('Fold ' + str(i+1) + ':')

        print('Preparing data')
        X, Y, x, y = prepare_data(k = i+1)
        print('done')

        print('Training')
        train(X, Y, x, y, fold = i+1)
        print('done')

        if i == 4:
            x_data = np.concatenate((X, x), axis=0)
            y_data = np.concatenate((Y, y), axis=0)
            final_clf = load_model("wavelet_model.hdf5")
            scores = final_clf.evaluate(x_data, y_data, BATCH_SIZE)
            print("Loss: ", scores[0])
            print("Accuracy: ", scores[1])

        del [X, Y, x, y]

    print('finished in ' + str(time.time()-t0) + ' secs')
