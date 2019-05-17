import pickle

from AbstractClass.StorageABC import Storage


class StoragePickle(Storage):
    def __init__(self, pathTodirectory):
        self.pathToDirectory = pathTodirectory

    def getItemById(self, id) -> dict:
        path = self.pathToDirectory + "\\" + id + ".pickle"
        with open(path, "rb") as readPickle:
            item = pickle.load(readPickle)
        return item

    def getItemsByFilter(self, filter: dict) -> list:
        pass

    def addItem(self, item: dict):
        pass

    def removeItem(self, id):
        pass

    def updateItem(self, id, updateData: dict):
        pass
