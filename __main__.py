from pprint import pprint as pp

from config import config
from developTools import toolsFile, researchData
from Class.Sample import SampleAdapter

import pandas as pn
import numpy as np

if __name__ == "__main__":
    debitDict = toolsFile.getDebitWell()

    dictDF = toolsFile.getDinamos()
    # toolsFile.saveToPickle(config.pathToPickle + "\\dinamos_test_debit_DICT.pickle", dictDF)

    # dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_test_debit_DICT.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getByDebitNorm()

    # data_TNSE = researchData.TNSE(sample)
    # toolsFile.saveToPickle(config.pathToPickle + "\\TNSE_state.pickle", data_TNSE)

    # data_TNSE = toolsFile.openFromPickle(config.pathToPickle + "\\TNSE_parts_2.pickle")
    # researchData.drawSemple(data_TNSE, path=config.pathToData + "\\TNSE_well.html", marker_in='marker_well')
    # researchData.drawSemple(data_TNSE, path=config.pathToData + "\\TNSE_state.html", marker_in='marker_state')
    # researchData.drawSemple(data_TNSE, path=config.pathToData + "\\TNSE_debit.html", marker_in='marker_debit')

    # debitDict = toolsFile.getDebitWell()
    # dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_debit_.pickle")
    # sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    # sample = sampleAdapter.getByWell()
    # pp(sample.loc[872:874])
    # new_sample = researchData.tree(sample, 9)

    # new_sample = researchData.k_means(sample.copy(True), 5)
    #
    # pp(pn.value_counts(new_sample['class']))
    # index_min = pn.value_counts(new_sample['class']).last_valid_index()
    # sample_1 = new_sample.loc[new_sample['class'] == index_min].reset_index(drop=True)
    #
    #
    # index_min = pn.value_counts(new_sample['class'])[~pn.value_counts(new_sample['class']).index.isin([index_min])].last_valid_index()
    # sample_2 = new_sample.loc[new_sample['class'] == index_min].reset_index(drop=True)
    #
    # pp(sample_1)
    # pp(sample_2)
    # sample_1.pop('marker_well')
    # sample_1.pop('marker_state')
    # sample_2.pop('marker_well')
    # sample_2.pop('marker_state')
    # sample_1.pop('class')
    # sample_2.pop('class')



    # for index in range(len(sample_1)):
    #     pp(index)
    #     researchData.drawDinamo(sample_1.loc[index])
    #
    # for index in range(len(sample_2)):
    #     pp(index)
    #     researchData.drawDinamo(sample_2.loc[index])

    # new_sample = researchData.TNSE(new_sample, "class")
    # researchData.drawSemple(new_sample, marker_in='class')

    # inputFeature = toolsFile.openFromPickle(config.pathToPickle + "\\inputFeature.pickle")
    # researchData.analizeFeatureInput(inputFeature, sample)
