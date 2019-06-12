from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Sample.Sample import SampleAdapter
from config import config
from filesTools import filesTools


def build_model(dim):
    model = keras.Sequential()

    # pp(isinstance(tf.keras.layers.Dense(1), tf.layers.Layer))

    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[dim]))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def train_model(data):
    train_labels = data.pop('marker_debit')

    model = build_model(len(data.keys()))

    EPOCHS = 1000

    model.fit(data, train_labels, epochs=EPOCHS, verbose=0)

    model.save_weights(config.pathToPickle + "\\keras_weight")
    # filesTools.saveToPickle(config.pathToPickle + "\\keras_model.pickle", model)


def draw_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure(figsize=(8, 12))

        plt.subplot(2, 1, 1)
        plt.xlabel('Эпоха')
        plt.ylabel('Среднее абсолютное отклонение')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label='Ошибка при обучении')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                 label='Ошибка при проверке')
        plt.ylim([0, 5])
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.xlabel('Эпоха')
        plt.ylabel('Среднеквадратическая ошибка')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
                 label='Ошибка при обучении')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                 label='Ошибка при проверке')
        plt.ylim([0, 20])
        plt.legend()
        plt.show()


def predict(data):
    data.pop('marker_debit')
    model = build_model(len(data.keys()))
    model.load_weights(config.pathToPickle + "\\keras_weight")
    predictions = model.predict(data)
    predictions = [item[0] for item in predictions]
    return np.array(predictions)


if __name__ == "__main__":
    debitDict = filesTools.getDebitWell()
    dictDF = filesTools.openFromPickle(config.pathToPickle + "\\dinamos_debit_.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getByDebitNorm()

    train_dataset = sample.sample(frac=0.8, random_state=42)
    test_dataset = sample.drop(train_dataset.index)
    test_labels = test_dataset['marker_debit']

    train_model(train_dataset.copy(True))

    predictions = predict(test_dataset.copy(True))
    error = abs(predictions - test_labels)

    print('Минимальная ошибка:', np.min(error))
    print('Максимальная ошибка:', np.max(error))
    print('Средняя ошибка:', np.mean(error))
    # print('стреднеквадратичное откланеие по всем прогнозам:', np.std(errors))
