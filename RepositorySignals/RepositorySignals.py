import os
import pickle

from config import config


class RepositorySignals(object):

    def __init__(self):
        self.__curentIndex = 0
        self.__filesNames = os.listdir(config.pathToPickleData)
        self.__maxIndex = len(os.listdir(config.pathToPickleData))
        pass

    def getNextSignal(self):
        if self.__curentIndex < self.__maxIndex:
            filename = self.__filesNames[self.__curentIndex]
            with open(config.pathToPickleData + "\\" + filename, "rb") as readPickle:
                signal = pickle.load(readPickle)

            result = {
                "status": 0,
                "filename": filename,
                "data": signal
            }

            self.__curentIndex += 1
            return result
        else:
            return {
                "status": -1,
                "filename": None,
                "data": None
            }

    def refreshIndex(self):
        self.__curentIndex = 0

    def getAmountSignal(self):
        return self.__maxIndex