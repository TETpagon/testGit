import xmltodict
import json
import pickle
import os
import pandas as pd
from pprint import pprint as pp

import re

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


def getDinamos():
    dinamos = {}
    dataCSV = getDinamosFromCSV()
    for well in dataCSV:
        dinamos[well] = {}
        for date in dataCSV[well]:
            dinamos[well][date] = {}
            for time in dataCSV[well][date]:
                df = dataCSV[well][date][time]
                if df.dtypes["force" and "position"] != 'object':
                    dinamosValues = getDinamosValues(df)
                    dinamos[well][date][time] = dinamosValues
    return dinamos


def getDinamosValues(df: pd.DataFrame):
    newDF = df.loc[~df['force'].isnull()].reset_index()
    newDF = newDF[['index', 'force', 'position']]
    indexes = list(newDF[newDF["index"].notnull()].index)
    indexes.append(newDF.index[-1])
    dinamos = []
    start = indexes[0]
    for index in indexes[1:]:
        dinamos.append(newDF.loc[start: index - 1].reset_index()[["force", "position"]])
        start = index
    return dinamos


def getDinamosFromCSV():
    data = {}
    for dirOilWell in os.listdir(config.pathToDirOilsWell):
        data[dirOilWell] = {}
        pathToDateDirs = config.pathToDirOilsWell + "\\" + dirOilWell
        dateDirs = os.listdir(pathToDateDirs)
        for dateDir in dateDirs:
            data[dirOilWell][dateDir] = {}
            pathToDinamos = pathToDateDirs + "\\" + dateDir
            dinamosCSV = os.listdir(pathToDinamos)
            for dinamo in dinamosCSV:
                pathToDinamoCSV = pathToDinamos + "\\" + dinamo
                df = pd.read_csv(pathToDinamoCSV, sep=";")
                data[dirOilWell][dateDir][dinamo[:-4]] = df
    return data


def getDebitWell():
    data = {}
    contentsExele = getDebitFromXML()
    for well in contentsExele:
        df = contentsExele[well][["Unnamed: 0", "Unnamed: 1", "Unnamed: 2", "Unnamed: 4"]].loc[6:]
        df.columns = ["Дата", "Время", "Интервал замера", "Дебит"]
        df = df.reset_index(drop=True)
        df = df.loc[df['Дата'].map(lambda x: True if re.search(r'\d\d\.02\.18', x) is not None else False)]
        data[well] = df
    return data


def getDebitFromXML():
    pathToDebit = {"Скважина 15795": config.pathToDebit_15795, "Скважина 18073": config.pathToDebit_18073,
                   "Скважина 30065": config.pathToDebit_30065}
    data = {}
    for well in pathToDebit:
        data[well] = pd.read_excel(pathToDebit[well])

    return data


def getListFile(path):
    pass


def saveToPickle(path, data):
    with open(path, "wb") as wr:
        pickle.dump(data, wr)


def openFromPickle(path):
    with open(path, "rb") as r:
        data = pickle.load(r)
    return data


if __name__ == "__main__":
    pass
