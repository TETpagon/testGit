from datetime import datetime, timedelta

import pandas as pn
from pprint import pprint as pp
import copy
import numpy as np

from config import config
from developTools import researchData, toolsFile


class SampleAdapter(object):
    def __init__(self, dictDF: dict = None, debitDict: dict = None):
        self.dictDF = dictDF
        self.debitDict = debitDict
        self.minAmountPoints = 576
        self.normDenamos = self._normalizeDinamos()

    def getByDebit(self):
        part = 2
        count = 0
        resultDinamos = pn.DataFrame()
        for well in self.dictDF:
            for date in self.dictDF[well]:
                try:

                    dinamos = []
                    for time in self.dictDF[well][date]:
                        dinamos += self.dictDF[well][date][time]
                    state = True
                    for dinamo in dinamos:
                        if np.max(dinamo['force']) - np.min(dinamo['force']) < 10000:
                            state = False
                            count += 1
                    if not state:
                        continue

                    if dinamos:
                        array = self._normalizeDinamo(dinamos[1]).values.ravel()
                        resultDinamo = np.split(array, part)[1]

                        array = self._normalizeDinamo(dinamos[-2]).values.ravel()
                        resultDinamo += np.split(array, part)[1]
                        df = pn.DataFrame(resultDinamo).transpose()
                        df['marker_well'] = ['Скважина 18073']
                        df['marker_state'] = ['bad']
                        resultDinamos = resultDinamos.append(df)
                except Exception as ex:
                    pp("{}   {}".format(well, date))
                    pp(self.dictDF[well][date])
                    raise ex
        pp(len(resultDinamos))
        return resultDinamos.reset_index(drop=True)

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
        part = 2
        count = 0
        resultDinamos = pn.DataFrame()
        for well in self.dictDF:
            for date in self.dictDF[well]:
                try:
                    dinamos = []
                    for time in self.dictDF[well][date]:
                        dinamos += self.dictDF[well][date][time]
                    state = True
                    for dinamo in dinamos:
                        if np.max(dinamo['force']) - np.min(dinamo['force']) < 10000:
                            state = False
                            count += 1
                    if not state:
                        continue

                    if dinamos and self.getDebitByDate(well, date):
                        debit = self.getDebitByDate(well, date)
                        # df = pn.DataFrame([self._normalizeDinamosValues(self._normalizeDinamo(dinamo)).values.ravel() for dinamo in dinamos])
                        df = pn.DataFrame([self._normalizeDinamo(dinamo).values.ravel() for dinamo in dinamos])
                        df['marker_debit'] = debit
                        resultDinamos = resultDinamos.append(df)
                except Exception as ex:
                    pp("{}   {}".format(well, date))
                    raise ex
        resultDinamos = resultDinamos.reset_index(drop=True)
        marker = resultDinamos['marker_debit']
        resultDinamos = resultDinamos.drop(['marker_debit'], axis=1)

        array2 = resultDinamos[resultDinamos.columns[1::2]].values

        meanA = np.mean(array2)
        stdA = np.std(array2)
        array2 -= meanA
        array2 /= stdA

        array1 = resultDinamos[resultDinamos.columns[0::2]].values

        meanA = np.mean(array1)
        stdA = np.std(array1)
        array1 -= meanA
        array1 /= stdA

        resultDinamos = pn.DataFrame(array1)
        resultDinamos = pn.concat([resultDinamos, pn.DataFrame(array2).rename(
            columns={item: str(int(item) + array1.shape[1]) for item in pn.DataFrame(array2).columns})], axis=1)
        resultDinamos['marker_debit'] = marker
        return resultDinamos.reset_index(drop=True)

    def getByDebitNormMean(self):
        part = 2
        count = 0
        resultDinamos = pn.DataFrame()
        for well in self.dictDF:
            for date in self.dictDF[well]:
                try:
                    dinamos = []
                    for time in self.dictDF[well][date]:
                        dinamos += self.dictDF[well][date][time]
                    state = True
                    for dinamo in dinamos:
                        if np.max(dinamo['force']) - np.min(dinamo['force']) < 5000:
                            state = False
                            count += 1
                    if not state:
                        continue
                    if dinamos and self.getDebitByDate(well, date):
                        debit = self.getDebitByDate(well, date)
                        df = pn.DataFrame([self._normalizeDinamosValues(self._normalizeDinamo(dinamo)).values.ravel() for dinamo in dinamos])
                        # df = pn.DataFrame([self._normalizeDinamo(dinamo).values.ravel() for dinamo in dinamos])
                        df['marker_debit'] = debit
                        resultDinamos = resultDinamos.append(pn.DataFrame(df.mean()).transpose())

                except Exception as ex:
                    pp("{}   {}".format(well, date))
                    raise ex
        resultDinamos = resultDinamos.reset_index(drop=True)
        marker = resultDinamos['marker_debit']
        resultDinamos = resultDinamos.drop(['marker_debit'], axis=1)

        array2 = resultDinamos[resultDinamos.columns[1::2]].values

        meanA = np.mean(array2)
        stdA = np.std(array2)
        array2 -= meanA
        array2 /= stdA

        array1 = resultDinamos[resultDinamos.columns[0::2]].values

        meanA = np.mean(array1)
        stdA = np.std(array1)
        array1 -= meanA
        array1 /= stdA

        resultDinamos = pn.DataFrame(array1)
        resultDinamos = pn.concat([resultDinamos, pn.DataFrame(array2).rename(
            columns={item: str(int(item) + array1.shape[1]) for item in pn.DataFrame(array2).columns})], axis=1)
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
        dfNew.sort_index()
        dfNew = dfNew.reset_index(drop=True)
        return dfNew.copy(True)

    def getDebitTime(self):
        result = pn.DataFrame()
        dictDebit = {}
        for well in self.debitDict:
            dictDebit[well] = []
            for item in self.debitDict[well].values:
                interval = datetime.strptime(item[2], "%H:%M")
                start = datetime.strptime(item[0] + " " + item[1], "%d.%m.%y %H:%M")
                end = start + timedelta(hours=interval.hour, minutes=interval.minute)
                dictDebit[well].append({
                    'start': start,
                    'end': end,
                    'debit': float(item[3]),
                })

            # for well in dictDebit:
            for debitTime in dictDebit[well]:
                dinamos = []
                matchStart = debitTime['start'].timestamp()
                matchEnd = debitTime['end'].timestamp()
                if debitDict['start'] != datetime(2018, 2, 7, 6, 5):
                    break
                for dateKey in self.dictDF[well]:
                    for timeKey in self.dictDF[well][dateKey]:
                        try:
                            dateR = dateKey.replace(".", "-")
                            dateMatch = datetime.strptime(dateR + " " + timeKey[3:], "%d-%m-%y %H-%M").timestamp()
                            diffStart = matchStart - dateMatch
                            diffEnd = matchEnd - dateMatch
                            if diffStart > -2700 and diffStart < -900:
                                dinamos.append(
                                    self._normalizeDinamo(self.dictDF[well][dateKey][timeKey][-1]).values.ravel())
                                pp((well, dateKey, timeKey, "-1"))
                            if diffStart < -2700 and diffEnd > -900:
                                pp((well, dateKey, timeKey, "all"))
                                dinamos += [self._normalizeDinamo(item).values.ravel() for item in
                                            self.dictDF[well][dateKey][timeKey]]
                            if diffEnd < -900 and diffEnd > -2700:
                                pp((well, dateKey, timeKey, "0"))
                                dinamos.append(
                                    self._normalizeDinamo(self.dictDF[well][dateKey][timeKey][0]).values.ravel())


                        # pp(diffStart)
                        # pp(diffEnd)
                        # if diffStart.days < 1 and diffStart.seconds < 1800:
                        #     if diffEnd.days < 1:
                        # print(diffStart, diffEnd)
                        # if diffStart.days < 1 and diffStart.seconds < 1800:
                        #     if diffEnd.days < 1 and diffEnd.seconds < 1800:
                        #         pp((dateKey, timeKey))
                        except Exception as ex:
                            pp((well, dateKey, timeKey))
                            raise ex

                # if dinamos and len(dinamos) > 7:
                if dinamos:
                    print()
                    pp(well)
                    pp(debitTime)
                    pp(debitTime['start'].timestamp())
                    pp(len(dinamos))
                    df = pn.DataFrame(dinamos)
                    df['marker_debit'] = debitTime['debit'] / len(dinamos)
                    result = result.append(df)
        input()
        resultDinamos = result.reset_index(drop=True)
        marker = resultDinamos['marker_debit']
        resultDinamos = resultDinamos.drop(['marker_debit'], axis=1)

        array2 = resultDinamos[resultDinamos.columns[1::2]].values

        meanA = np.mean(array2)
        stdA = np.std(array2)
        array2 -= meanA
        array2 /= stdA

        array1 = resultDinamos[resultDinamos.columns[0::2]].values

        meanA = np.mean(array1)
        stdA = np.std(array1)
        array1 -= meanA
        array1 /= stdA

        resultDinamos = pn.DataFrame(array1)
        resultDinamos = pn.concat([resultDinamos, pn.DataFrame(array2).rename(
            columns={item: str(int(item) + array1.shape[1]) for item in pn.DataFrame(array2).columns})], axis=1)
        resultDinamos['marker_debit'] = marker
        input('stop')
        return resultDinamos.reset_index(drop=True)


if __name__ == "__main__":
    debitDict = toolsFile.getDebitWell()
    dictDF = toolsFile.getDinamosDebit()
    toolsFile.saveToPickle(config.pathToPickle + "\\dinamos_debit_DICT.pickle", dictDF)
    # dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_debit_DICT.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    sample = sampleAdapter.getDebitTime()
    pp(sample)
