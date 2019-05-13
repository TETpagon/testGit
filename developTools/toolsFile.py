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



