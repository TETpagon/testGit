from sklearn.ensemble import RandomForestRegressor

from Class.Sample import SampleAdapter
from config import config
from developTools import toolsFile

import numpy as np


def train_model(sample):
    sample = sample.sample(len(sample))

    train_dataset = sample.sample(len(sample))

    train_labels = train_dataset.pop('marker_debit')

    model = RandomForestRegressor(n_estimators=1000, random_state=10, n_jobs=-1)
    model.fit(train_dataset.values, train_labels.values)

    toolsFile.saveToPickle(config.pathToPickle + "\\randomForest_model.pickle", model)


def predict(data):
    model = toolsFile.openFromPickle(config.pathToPickle + "\\randomForest_model.pickle")
    data.pop('marker_debit')
    predictions = model.predict(data.values)

    return predictions


if __name__ == "__main__":
    debitDict = toolsFile.getDebitWell()
    dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_debit_.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getByDebitNorm()

    train_sample = sample.sample(frac=0.8)
    test_sample = sample.drop(train_sample.index)

    train_model(train_sample.copy(True))

    predictions = predict(test_sample.copy(True))

    errors = abs(predictions - test_sample['marker_debit'].values)

    print('Минимальная ошибка:', np.min(errors))
    print('Максимальная ошибка:', np.max(errors))
    print('Средняя ошибка:', np.mean(errors))
    # print('стреднеквадратичное откланеие по всем прогнозам:', np.std(errors))
