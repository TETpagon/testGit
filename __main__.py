import json
from pprint import pprint as pp
import os
import xmltodict
import pickle

from config import config
from developTools import researchFilesSignals
from developTools import toolsFile
from RepositorySignals.RepositorySignals import RepositorySignals

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

    repo = RepositorySignals()
    for index in range(repo.getAmountSignal()):
        signal = repo.getNextSignal()




