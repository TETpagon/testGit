from analisisTools.researchData import TNSE, drawSemple, k_means, tree
from config import config
from filesTools import filesTools
from Sample.Sample import SampleAdapter
import pandas as pn

if __name__ == "__main__":
    debitDict = filesTools.getDebitWell()

    dictDF = filesTools.openFromPickle(config.pathToPickle + "\\dinamos_debit_.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getByDebitNorm()

    data_TNSE = TNSE(sample, marker='marker_well')
    drawSemple(data_TNSE, path=config.pathToData + "\\TNSE_debit.html", marker_in='marker_debit')

    debitDict = filesTools.getDebitWell()
    dictDF = filesTools.openFromPickle(config.pathToPickle + "\\dinamos_debit_.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getByWell()

    new_sample = tree(sample, 9)
    new_sample = k_means(sample.copy(True), 5)

    index_min = pn.value_counts(new_sample['class']).last_valid_index()
    sample_1 = new_sample.loc[new_sample['class'] == index_min].reset_index(drop=True)
    index_min = pn.value_counts(new_sample['class'])[
        ~pn.value_counts(new_sample['class']).index.isin([index_min])].last_valid_index()
    sample_2 = new_sample.loc[new_sample['class'] == index_min].reset_index(drop=True)

    new_sample = TNSE(new_sample, "class")
    drawSemple(new_sample, marker_in='class')
