import pandas as pn
from pprint import pprint as pp
import copy
import numpy as np


class SampleAdapter(object):
    """
        Класс отвечает за работу с динамограмми и дебитом
    """

    def __init__(self, dictDF: dict = None, debitDict: dict = None):
        self.dictDF = dictDF
        self.debitDict = debitDict
        self.minAmountPoints = 576  # минимальное количество значений в динамограмме (получено экспереминтально)

    def getDebitByDate(self, well, date):
        indexes = {
            '': 0,
            ' — копия': 1,
            ' — копия (2)': 2,
            ' — копия (3)': 3,
        }

        dateDebit = str(date[:8]).replace("-", ".")
        # pp(date)
        debit = self.debitDict[well].loc[self.debitDict[well]['Дата'] == dateDebit]['Дебит'].reset_index(drop=True)
        # pp(pn.DataFrame.isnull)
        if not debit.empty:
            resultDebit = float(debit.loc[indexes[date[8:]]])
        else:
            resultDebit = None

        return resultDebit

    def getByDebitNorm(self):
        """
            Функция возвращает динамограммы в соотвествие которым поставлен дебит

            Возвращает:
                pandas.DataFrame
        """
        count = 0
        resultDinamos = pn.DataFrame()
        for well in self.dictDF:
            for date in self.dictDF[well]:
                dinamos = []
                for time in self.dictDF[well][date]:
                    dinamos += self.dictDF[well][date][time]
                state = True
                for dinamo in dinamos:
                    if np.max(dinamo['force']) - np.min(dinamo['force']) < 3000:
                        state = False
                        count += 1
                if not state:
                    continue

                if dinamos and self.getDebitByDate(well, date):
                    debit = self.getDebitByDate(well, date)
                    df = pn.DataFrame([self._normalizeDinamo(dinamo).values.ravel() for dinamo in dinamos])
                    df['marker_debit'] = debit
                    resultDinamos = resultDinamos.append(df)
        resultDinamos = resultDinamos.reset_index(drop=True)
        marker = resultDinamos['marker_debit']
        resultDinamos = resultDinamos.drop(['marker_debit'], axis=1)

        array1 = resultDinamos[resultDinamos.columns[0::2]]

        meanA = np.mean(array1)
        stdA = np.std(array1)
        array1 -= meanA
        array1 /= stdA

        array2 = resultDinamos[resultDinamos.columns[1::2]]

        meanA = np.mean(array2)
        stdA = np.std(array2)
        array2 -= meanA
        array2 /= stdA

        resultDinamos = pn.concat([array1, array2], axis=1)
        cols = resultDinamos.columns.tolist()
        cols = sorted(cols)
        resultDinamos = resultDinamos[cols]
        resultDinamos['marker_debit'] = marker
        return resultDinamos.reset_index(drop=True)

    def getStateWell(self):
        dinamosWells = pn.DataFrame()
        count = 0
        for well in self.dictDF:
            for date in self.dictDF[well]:
                for time in self.dictDF[well][date]:
                    for dinamo in self.dictDF[well][date][time]:
                        if np.max(dinamo['force']) - np.min(dinamo['force']) < 10000:
                            count += 1
                            state = 'bad'
                        else:
                            state = 'good'
                        array = self._normalizeDinamo(dinamo).values.ravel()
                        df = pn.DataFrame(array)
                        df = df.transpose()
                        df['marker'] = [state]
                        dinamosWells = dinamosWells.append(df)
        pp(count)
        return dinamosWells.reset_index(drop=True)

    def getByWell(self):
        """
            Функция возвращает динамограммы, в соответсвие котормы ставиться вышка

            Возвращает:
                pandas.DataFrame
        """
        dinamosWells = pn.DataFrame()
        for well in self.dictDF:
            for date in self.dictDF[well]:
                for time in self.dictDF[well][date]:
                    for dinamo in self.dictDF[well][date][time]:
                        if np.max(dinamo['force']) - np.min(dinamo['force']) < 10000:
                            state = 'bad'
                        else:
                            state = 'good'
                        array = self._normalizeDinamo(dinamo).values.ravel()
                        df = pn.DataFrame(array)
                        df = df.transpose()
                        df['marker_well'] = [well]
                        df['marker_state'] = [state]
                        dinamosWells = dinamosWells.append(df)
        return dinamosWells.reset_index(drop=True)

    def getByWellNorm(self):
        dinamosWells = pn.DataFrame()
        for well in self.dictDF:
            for date in self.dictDF[well]:
                for time in self.dictDF[well][date]:
                    for dinamo in self.dictDF[well][date][time]:
                        if np.max(dinamo['force']) - np.min(dinamo['force']) < 10000:
                            state = 'bad'
                        else:
                            state = 'good'
                        array = self._normalizeDinamo(dinamo).values.ravel()
                        array = array - np.min(array)
                        array = array / np.max(array)
                        df = pn.DataFrame(array)
                        df = df.transpose()
                        df['marker_well'] = [well]
                        df['marker_state'] = [state]
                        dinamosWells = dinamosWells.append(df)

        return dinamosWells.reset_index(drop=True)

    def getByParts(self, part_in=9):
        if self.minAmountPoints % part_in == 0:
            part = part_in
        else:
            part = 1

        dinamosParts = pn.DataFrame()
        for well in self.dictDF:
            for date in self.dictDF[well]:
                for time in self.dictDF[well][date]:
                    for dinamo in self.dictDF[well][date][time]:
                        array = self._normalizeDinamo(dinamo).values.ravel()
                        df = pn.DataFrame(
                            [list(item) + list([part]) for part, item in enumerate(np.split(array, part))])
                        df = df.rename(columns={df.columns[-1]: 'marker'})
                        dinamosParts = dinamosParts.append(df)
        return dinamosParts.reset_index(drop=True)

    def _normalizeDinamos(self):
        normDinamos = {}
        for well in self.dictDF:
            normDinamos[well] = {}
            for date in self.dictDF[well]:
                normDinamos[well][date] = {}
                for time in self.dictDF[well][date]:
                    normDinamos[well][date][time] = []
                    for dinamo in self.dictDF[well][date][time]:
                        normDinamos[well][date][time].append(self._normalizeDinamo(dinamo))
        return normDinamos

    def _normalizeDinamosValues(self, dinamos):
        d = copy.deepcopy(dinamos)
        d = d - np.min(d)
        d = d / np.max(d)
        return d

    def _normalizeDinamo(self, dinamoDF):
        dfNew = dinamoDF.copy(True)
        start = len(dfNew) - self.minAmountPoints
        dfNew = dfNew.loc[start:]
        dfNew = dfNew.reset_index(drop=True)
        return dfNew.copy(True)


if __name__ == "__main__":
    pass
