from pprint import pprint as pp

from config import config
from developTools import toolsFile, researchData, SOM
from Class.RepositorySignals import RepositorySignals
from Class.Sample import SampleAdapter

if __name__ == "__main__":
    # dataFromXML = researchFilesSignals.getDataFromFilesSignals()
    # researchFilesSignals.writeToPickle(config.pathToDataFromXML, dataFromXML)
    #
    # # dataFromXML = researchFilesSignals.readeFromPickle(config.pathToDataFromXML)
    #
    # valuesSignals = researchFilesSignals.getValuesSignal(dataFromXML)
    # del(dataFromXML)
    # researchFilesSignals.writeToPickle(config.pathToValuesSignals, valuesSignals)
    #
    # amountValuesInSignal = researchFilesSignals.getAmountValuesInSignal(valuesSignals)
    # del(valuesSignals)
    # researchFilesSignals.writeToPickle(config.pathToAmountValuesInSignal, amountValuesInSignal)
    #
    # minMaxAmountValueInSignal = researchFilesSignals.getMinMaxAmountValueInSignal(amountValuesInSignal)
    # del(amountValuesInSignal)
    # researchFilesSignals.writeToPickle(config.pathToMinMaxAmountValueInSignal, minMaxAmountValueInSignal)

    # toolsFile.saveXmlFilesToPickle()

    # repo = RepositorySignals()
    # values = {}
    # for index in range(repo.getAmountSignal()):
    #     signal = repo.getNextSignal()
    #     value = repo.getValuesSignal(signal)
    #     if (value['fTickDuration'] or value['fTickDuration-Data']) and value['fAcceleration'] and value['fForce'] and \
    #             value['fPosition']:
    #         values['values-' + signal['filename']] = value
    #
    #     # if value['fAcceleration'] is not None and value['fForce'] is not None:
    #     #     values['values-' + signal['filename']] = value
    # toolsFile.saveValuesSignalToPickle(values)

    # listDF, dictDF = researchData.getSample()

    ####################################################################################################################

    debitDict = toolsFile.getDebitWell()

    dictDF = toolsFile.getDinamosDebit()
    # toolsFile.saveToPickle(config.pathToPickle + "\\dinamos_debit_DICT.pickle", dictDF)

    dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_debit_DICT.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    # sample = sampleAdapter.getByParts(2)
    # sample = sampleAdapter.getByDebitNorm()
    sample = sampleAdapter.getByDebitNormMean()
    pp(sample.shape)

    # data_TNSE = researchData.TNSE(sample)
    # toolsFile.saveToPickle(config.pathToPickle + "\\TNSE_state.pickle", data_TNSE)

    # data_TNSE = toolsFile.openFromPickle(config.pathToPickle + "\\TNSE_parts_2.pickle")
    # researchData.drawSemple(data_TNSE, path=config.pathToData + "\\TNSE_well.html", marker_in='marker_well')
    # researchData.drawSemple(data_TNSE, path=config.pathToData + "\\TNSE_state.html", marker_in='marker_state')
    # researchData.drawSemple(data_TNSE, path=config.pathToData + "\\TNSE_debit.html", marker_in='marker_debit')

    # listDF = researchData.convertDFTOArray2D(listDF)
    # listDF = researchData.convertDFTOArray64D_9(listDF)

    # researchData.drawDinamo(listDF[0])

    # researchData.PCA(listDF)
    # researchData.TNSE(listDF)
    # researchData.DBSCAN(listDF)
    # researchData.spectr(listDF)
    # researchData.andrews(listDF)
    # result = SOM.test_som_with_color_data(listDF)

    # toolsFile.saveToPickle(config.pathToPickle+"\\som.pickle", result)
    # som_W = toolsFile.openFromPickle(config.pathToPickle + "\\som.pickle")
    # SOM.drawSom(som_W)
