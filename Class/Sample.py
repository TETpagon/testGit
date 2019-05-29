import pandas as pn
from pprint import pprint as pp
import copy
import numpy as np

from developTools import researchData


class SampleAdapter(object):
    def __init__(self, dictDF: dict = None, debitDict: dict = None):
        self.dictDF = dictDF
        self.minAmountPoints = 576
        self.normDenamos = self._normalizeDinamos()

    def getByDebit(self):
        pass

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
                        df['marker_state'] = [well]
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
        normDenamos = {}
        for well in self.dictDF:
            normDenamos[well] = {}
            for date in self.dictDF[well]:
                normDenamos[well][date] = {}
                for time in self.dictDF[well][date]:
                    normDenamos[well][date][time] = []
                    for dinamo in self.dictDF[well][date][time]:
                        normDenamos[well][date][time].append(self._normalizeDinamo(dinamo))
        return normDenamos

    def _normalizeDinamo(self, dinamoDF):
        dfNew = dinamoDF.copy(True)
        start = len(dfNew) - self.minAmountPoints
        dfNew = dfNew.loc[start:]
        dfNew.sort_index()
        dfNew = dfNew.reset_index(drop=True)
        return dfNew.copy(True)
