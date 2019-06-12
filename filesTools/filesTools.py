from io import StringIO
import chardet
import pickle
import os
import pandas as pd
from pprint import pprint as pp

import re

from config import config


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


if __name__ == "__main__":
    pass
