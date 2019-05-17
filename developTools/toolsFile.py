import xmltodict
import json
import pickle
import os
from config import config


def saveXmlFilesToPickle():
    filesNames = os.listdir(config.pathToDataXML)
    amountFiles = len(filesNames)
    for index, fileName in enumerate(filesNames):
        with open(config.pathToDataXML + "\\" + fileName) as readFile:
            xml = readFile.read()
        doc = xmltodict.parse(xml[3:])
        dictSignal = json.loads(json.dumps(doc))

        with open(config.pathToPickleData + "\\" + str(fileName[:-4]) + ".pickle", "wb")as writeFile:
            pickle.dump(dictSignal, writeFile)
        if index % 100 == 0:
            print("Обработано {} из {} файлов.".format(index, amountFiles))


def saveValuesSignalToPickle(valuesSignals: dict):
    for filename in valuesSignals:
        with open(config.pathToValuesSignals + "\\" + filename, "wb")as writeFile:
            pickle.dump(valuesSignals[filename], writeFile)


def saveToPickle(path, data):
    with open(path, "wb") as wr:
        pickle.dump(data, wr)


def openFromPickle(path):
    with open(path, "rb") as r:
        data = pickle.load(r)
    return data
