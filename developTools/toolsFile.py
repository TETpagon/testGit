from io import StringIO

import chardet
import xmltodict
import json
import pickle
import os
import pandas as pd
from pprint import pprint as pp
from datetime import datetime
from copy import deepcopy
import shutil

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
    """
        Получение значений динамограмм 
        
        Возвращает:
            словарь, содержимое которого {Скважина -> Дата -> Время -> Динамогрммы}
    """
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


def getDinamosDebit():
    dinamos = {}
    dataCSV = getDinamosFromCSV(config.pathToDirOilsWellOrder)
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
    """
        Функция среди данных выбирает только нужные значения динаммограмм 
        
        Возвращает:
            Список динамограмм
    """
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


def getDinamosFromCSV(path=config.pathToDirOilsWell):
    """
        Фнукция чтения данных из файла CSV
        
        Параметры:
            path - путь, по  которму лежит файл
        
        Вовращает:
            словарь, содержимое которого которого {Скважина -> Дата -> Время -> динамограммы} 
    """
    data = {}
    for dirOilWell in os.listdir(path):
        data[dirOilWell] = {}
        pathToDateDirs = path + "\\" + dirOilWell
        dateDirs = os.listdir(pathToDateDirs)
        for dateDir in dateDirs:
            data[dirOilWell][dateDir] = {}
            pathToDinamos = pathToDateDirs + "\\" + dateDir
            dinamosCSV = os.listdir(pathToDinamos)
            for dinamo in dinamosCSV:
                try:
                    pathToDinamoCSV = pathToDinamos + "\\" + dinamo

                    with open(pathToDinamoCSV, "r") as file:
                        dataCSV = file.read()
                    df = pd.read_csv(StringIO(dataCSV), sep=";")
                    data[dirOilWell][dateDir][dinamo[:-4]] = df
                except Exception as ex:
                    with open(pathToDinamos + "\\" + dinamo, "rb") as file:
                        date = file.read()
                        result = chardet.detect(date)
                        charenc = result['encoding']
                        pp(charenc)
                    pp(pathToDinamos + "\\" + dinamo)
                    raise ex
    return data


def getDebitWell() -> dict:
    """
        Функция получения данных о дебите.
        
        Вовращает:
            словарь, содержимое которого данные о дебите по каждой вышке
    """
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
    """
        Чтение данных о дебите из файла Excel
        
        Возвращает:
            словарь с данными о дебите по каждой вышке
    """
    pathToDebit = {"Скважина 15795": config.pathToDebit_15795, "Скважина 18073": config.pathToDebit_18073,
                   "Скважина 30065": config.pathToDebit_30065}
    data = {}
    for well in pathToDebit:
        data[well] = pd.read_excel(pathToDebit[well])

    return data


def getListFile(path):
    pass


def saveToPickle(path, data):
    """
        Функция сохраняет данные на диске в формате pickle
        
        Параметры:
            path - путь, по которму сохраняются данные
            data - данные для сохранения
    """
    with open(path, "wb") as wr:
        pickle.dump(data, wr)


def openFromPickle(path):
    """
        Функция загружает данные из файла pickle
        
        Параметры: 
            path - путь, по кторому данные лежат
        
        Возвращает: 
            данные
    """
    with open(path, "rb") as r:
        data = pickle.load(r)
    return data


def createSempleDebit():
    pathToWell = config.pathToDirOilsWell
    wells = os.listdir(pathToWell)
    fileDateTime = {}
    for well in wells:
        fileDateTime[well] = {}
        pathToDate = pathToWell + "\\" + well
        dates = os.listdir(pathToDate)
        for date in dates:
            fileDateTime[well][date] = {}
            pathToTime = pathToDate + "\\" + date
            times = os.listdir(pathToTime)
            for file in times:
                pathToFile = pathToTime + "\\" + file
                fileDateTime[well][date][file] = os.stat(pathToFile).st_mtime

    sortFile = {}
    for well in fileDateTime:
        sortFile[well] = {}
        for date in fileDateTime[well]:
            sortFile[well][date] = {}
            x = deepcopy(fileDateTime[well][date])
            sorted_x = sorted(x.items(), key=lambda kv: kv[1])
            for index, item in enumerate(sorted_x):
                sortFile[well][date][item[0]] = index

    resultFile = {}

    for well in wells:
        resultFile[well] = {}
        pathToDate = pathToWell + "\\" + well
        if not os.path.exists(pathToDate.replace('все Динамограммы', 'отсортированные Динамограммы')):
            os.mkdir(pathToDate.replace('все Динамограммы', 'отсортированные Динамограммы'))
        dates = os.listdir(pathToDate)
        for date in dates:
            resultFile[well][date] = {}
            pathToTime = pathToDate + "\\" + date
            if not os.path.exists(pathToTime.replace('все Динамограммы', 'отсортированные Динамограммы')):
                os.mkdir(pathToTime.replace('все Динамограммы', 'отсортированные Динамограммы'))
            times = os.listdir(pathToTime)
            for file in times:
                pathToFile = pathToTime + "\\" + file
                pathMove = pathToTime
                pathMove = pathMove.replace('все Динамограммы', 'отсортированные Динамограммы')
                index = sortFile[well][date][file]
                index = str(index) if index // 10 != 0 else '0' + str(index)
                pathMove = pathMove + "\\{}_".format(index) + file
                shutil.copy2(pathToFile, pathMove)

    pathToWell = 'F:\\Projects\\Data\\Данные по скважинам\\отсортированные Динамограммы'
    wells = os.listdir(pathToWell)

    for well in wells:
        resultFile[well] = {}
        pathToDate = pathToWell + "\\" + well
        if not os.path.exists(
                pathToDate.replace('отсортированные Динамограммы', 'отсортированные отформатированные Динамограммы')):
            os.mkdir(
                pathToDate.replace('отсортированные Динамограммы', 'отсортированные отформатированные Динамограммы'))
        dates = os.listdir(pathToDate)
        for date in dates:
            base = ''
            resultFile[well][date] = {}
            pathToTime = pathToDate + "\\" + date
            if not os.path.exists(pathToTime.replace('отсортированные Динамограммы',
                                                     'отсортированные отформатированные Динамограммы')):
                os.mkdir(pathToTime.replace('отсортированные Динамограммы',
                                            'отсортированные отформатированные Динамограммы'))
            times = os.listdir(pathToTime)
            for file in times:
                pathToFile = pathToTime + "\\" + file
                pathMove = pathToTime
                pathMove = pathMove.replace('отсортированные Динамограммы',
                                            'отсортированные отформатированные Динамограммы')
                base, newfile = norm(base, file)
                pathMove = pathMove + "\\" + newfile
                shutil.copy2(pathToFile, pathMove)


def norm(base, file):
    newbase = file
    newfile = file
    if 'signal' in file and 'signal' not in base and base != '':
        try:
            parts = base.split("_")
            parts = parts[1].split(".")
            ex = parts[1]
            time = parts[0].split("-")
            h = int(time[0]) + 1
            # pp((base, file, h))
            h = 0 if h >= 24 else h
            h = str(h) if h // 10 != 0 else '0' + str(h)
            index = file.split("_")[0]
            newfile = "{}_{}-{}.{}".format(index, str(h), time[1], ex)
            newbase = newfile

        except Exception as ex:
            pp(base)
            pp(file)
            raise ex

    return newbase, newfile


if __name__ == "__main__":
    createSempleDebit()
