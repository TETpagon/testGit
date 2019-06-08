from pprint import pprint as pp

from config import config
from developTools import toolsFile, researchData
from Class.Sample import SampleAdapter

if __name__ == "__main__":
    # debitDict = toolsFile.getDebitWell()

    # dictDF = toolsFile.getDinamos()
    # toolsFile.saveToPickle(config.pathToPickle + "\\dinamos_test_debit_DICT.pickle", dictDF)

    # dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_test_debit_DICT.pickle")
    # sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    # sample = sampleAdapter.getByParts(2)
    # sample = sampleAdapter.getByDebitNorm()
    # sample = sampleAdapter.getByDebitNormMean()
    # pp(sample.shape)

    # data_TNSE = researchData.TNSE(sample)
    # toolsFile.saveToPickle(config.pathToPickle + "\\TNSE_state.pickle", data_TNSE)

    # data_TNSE = toolsFile.openFromPickle(config.pathToPickle + "\\TNSE_parts_2.pickle")
    # researchData.drawSemple(data_TNSE, path=config.pathToData + "\\TNSE_well.html", marker_in='marker_well')
    # researchData.drawSemple(data_TNSE, path=config.pathToData + "\\TNSE_state.html", marker_in='marker_state')
    # researchData.drawSemple(data_TNSE, path=config.pathToData + "\\TNSE_debit.html", marker_in='marker_debit')

    debitDict = toolsFile.getDebitWell()
    dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_debit_.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getByDebitNorm()

    pp(sample)

    # inputFeature = toolsFile.openFromPickle(config.pathToPickle + "\\inputFeature.pickle")
    # researchData.analizeFeatureInput(inputFeature, sample)
