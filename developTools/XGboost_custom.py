import xgboost

from Class.Sample import SampleAdapter
from config import config
from developTools import toolsFile
import numpy as np

if __name__ == "__main__":
    debitDict = toolsFile.getDebitWell()

    dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_debit_DICT.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getByDebitNorm()
    sample = sample.sample(len(sample))

    y = sample['marker_debit'].values
    sample = sample.drop(['marker_debit'], axis=1)
    X = sample.values

    xgb = xgboost.XGBRegressor(n_estimators=3000, learning_rate=0.08, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=300, n_jobs=-1)
    xgb.fit(X[:900], y[:900])
    predictions = xgb.predict(X[900:])
    errors = abs(predictions - y[900:])

    print('Min Absolute Error:', np.min(errors), 'degrees.')
    print('Max Absolute Error:', np.max(errors), 'degrees.')
    print('Mean Absolute Error:', np.mean(errors), 'degrees.')
    print('Std Absolute Error:', np.std(errors), 'degrees.')