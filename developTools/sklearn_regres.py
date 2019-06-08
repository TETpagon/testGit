from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

from Class.Sample import SampleAdapter
from config import config
from developTools import toolsFile
from sklearn.model_selection import GridSearchCV

import numpy as np

from pprint import pprint as pp

if __name__ == "__main__":
    debitDict = toolsFile.getDebitWell()
    dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_debit_.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getByDebitNorm()
    sample = sample.sample(len(sample))

    train_dataset = sample.sample(frac=0.8, random_state=0)
    test_dataset = sample.drop(train_dataset.index)

    train_labels = train_dataset.pop('marker_debit')
    test_labels = test_dataset.pop('marker_debit')

    model = RandomForestRegressor(n_estimators=1000, random_state=10, n_jobs=-1)

    # model.fit(X[:900], y[:900])
    # predictions = model.predict(X[900:])
    # errors = abs(predictions - y[900:])

    # param_grid = {
    #     'n_estimators': [1000],
    #     'max_features': ['sqrt'],
    # }
    #
    # CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=3)
    # CV_rfc.fit(train_dataset.values, train_labels.values)
    # predictions = CV_rfc.predict(test_dataset.values)

    model.fit(train_dataset.values, train_labels.values)
    predictions = model.predict(test_dataset.values)

    # pp(model)

    errors = abs(predictions - test_labels.values)

    # pp(CV_rfc.best_params_)

    print('Минимальная ошибка:', np.min(errors))
    print('Максимальная ошибка:', np.max(errors))
    print('Средняя ошибка:', np.mean(errors))
    # print('стреднеквадратичное откланеие по всем прогнозам:', np.std(errors))

    toolsFile.saveToPickle(config.pathToPickle + "\\inputFeature.pickle", model.feature_importances_.tolist())
