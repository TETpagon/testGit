from abc import ABCMeta, abstractmethod, abstractproperty


class Storage(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getItemById(self, id) -> dict:
        '''
        :param id:
        :return dict:
        '''
        pass

    @abstractmethod
    def getItemsByFilter(self, filter: dict) -> list:
        '''
        :param filter:
        :return list:
        '''
        pass

    @abstractmethod
    def addItem(self, item: dict):
        '''
        :param item:
        :return:
        '''
        pass

    @abstractmethod
    def removeItem(self, id):
        '''
        :param id:
        :return:
        '''
        pass

    @abstractmethod
    def updateItem(self, id, updateData: dict):
        '''
        :param id:
        :param updateData:
        :return:
        '''
        pass
