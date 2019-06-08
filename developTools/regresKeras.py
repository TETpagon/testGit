from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pprint import pprint as pp
import numpy as np


from Class.Sample import SampleAdapter
from config import config
from developTools import toolsFile

if __name__ == "__main__":
    debitDict = toolsFile.getDebitWell()

    dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_debit_.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getByDebitNorm()

    train_dataset = sample.sample(frac=0.8, random_state=0)
    test_dataset = sample.drop(train_dataset.index)

    train_labels = train_dataset.pop('marker_debit')
    test_labels = test_dataset.pop('marker_debit')


    def build_model():
        model = keras.Sequential()

        # pp(isinstance(tf.keras.layers.Dense(1), tf.layers.Layer))

        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]))
        model.add(tf.keras.layers.Dense(1))

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model


    print(tf.VERSION)
    print(tf.keras.__version__)

    model = build_model()
    model.summary()

    EPOCHS = 1000

    history = model.fit(train_dataset, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0)

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


    # plot_history(history)

    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)

    print("Среднее абсолютное отклонение на проверочных данных: {:5.4f}".format(loss))

    test_predictions = model.predict(test_dataset).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    # plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")

    # plt.show()
    error = abs(test_predictions - test_labels)

    print('Минимальная ошибка:', np.min(error))
    print('Максимальная ошибка:', np.max(error))
    print('Средняя ошибка:', np.mean(error))
    # print('стреднеквадратичное откланеие по всем прогнозам:', np.std(errors))