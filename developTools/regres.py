from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import numpy as np

from Class.Sample import SampleAdapter
from config import config
from developTools import toolsFile

from pprint import pprint as pp

if __name__ == "__main__":
    debitDict = toolsFile.getDebitWell()
    dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_debit_DICT.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getByDebitNorm()
    sample = sample.sample(len(sample))

    y = sample['marker_debit'].values
    sample = sample.drop(['marker_debit'], axis=1)
    X = sample.values

    rfr = RandomForestRegressor(n_estimators=3000, random_state=300, max_features='sqrt', n_jobs=-1)  # случайный лес
    svr = SVR(kernel='linear')  # метод опорных векторов с линейным ядром

    rfr.fit(X[:900], y[:900])
    svr.fit(X[:900], y[:900])
    lab_enc = preprocessing.LabelEncoder()
    training_scores_encoded = lab_enc.fit_transform(y[:900])

    predictions = rfr.predict(X[900:])
    errors = abs(predictions - y[900:])

    print('RandomForestRegressor min :', np.min(errors), 'error.')
    print('RandomForestRegressor max :', np.max(errors), 'error.')
    print('RandomForestRegressor mean :', np.mean(errors), 'error.')
    print('RandomForestRegressor std:', np.std(errors), 'error.')
    print()

    predictions = svr.predict(X[900:])
    errors = abs(predictions - y[900:])

    print('SVR min :', np.min(errors), 'error.')
    print('SVR max :', np.max(errors), 'error.')
    print('SVR mean :', np.mean(errors), 'error.')
    print('SVR std:', np.std(errors), 'error.')
    print()
