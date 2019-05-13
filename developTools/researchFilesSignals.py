import json
from pprint import pprint as pp
import os
import xmltodict
import pickle
from config import config


def getDataFromFilesSignals():
    filesNames = os.listdir(config.pathToDataXML)
    signals = {}

    for fileName in filesNames:
        with open(config.pathToDataXML + "\\" + fileName) as readFile:
            xml = readFile.read()

        doc = xmltodict.parse(xml[3:])
        doc = json.loads(json.dumps(doc))
        if fileName not in signals.keys():
            signals[fileName[:-4]] = doc
        else:
            raise Exception("Сигнал с таким именем именем уже существует!!!")
    return signals


def getValuesSignal(signals):
    values = {}
    for key in signals:
        try:
            values[key] = {
                'fTickDuration': signals[key]['obj']['Measurement']['@fAcceleration'].split(" "),
                'fAcceleration': signals[key]['obj']['Measurement']['@fAcceleration'].split(" "),
                'fForce': signals[key]['obj']['Measurement']['@fAcceleration'].split(" "),
                'fPosition': signals[key]['obj']['Measurement']['@fAcceleration'].split(" "),
            }
        except:
            pass
    return values


def getAmountValuesInSignal(valuesSignals):
    amountValues = {}
    for key in valuesSignals:
        amountValues[key] = {
            'fTickDuration': len(valuesSignals['fTickDuration']),
            'fAcceleration': len(valuesSignals['fAcceleration']),
            'fForce': len(valuesSignals['fForce']),
            'fPosition': len(valuesSignals['fPosition']),
        }
    return amountValues


def getMinMaxAmountValueInSignal(amountValuesInSignal):
    minMaxAmountValues = {
        'fTickDuration': {
            'min': 99999999, 'max': -9999999
        },
        'fAcceleration': {
            'min': 99999999, 'max': -9999999
        },
        'fForce': {
            'min': 99999999, 'max': -9999999
        },
        'fPosition': {
            'min': 99999999, 'max': -9999999
        },
    }
    for keys in amountValuesInSignal:
        if minMaxAmountValues['fTickDuration']['max'] < amountValuesInSignal[keys]['fTickDuration']:
            minMaxAmountValues['fTickDuration']['max'] = amountValuesInSignal[keys]['fTickDuration']
        if minMaxAmountValues['fTickDuration']['min'] > amountValuesInSignal[keys]['fTickDuration']:
            minMaxAmountValues['fTickDuration']['min'] = amountValuesInSignal[keys]['fTickDuration']

        if minMaxAmountValues['fAcceleration']['max'] < amountValuesInSignal[keys]['fAcceleration']:
            minMaxAmountValues['fAcceleration']['max'] = amountValuesInSignal[keys]['fAcceleration']
        if minMaxAmountValues['fAcceleration']['min'] > amountValuesInSignal[keys]['fAcceleration']:
            minMaxAmountValues['fAcceleration']['min'] = amountValuesInSignal[keys]['fAcceleration']

        if minMaxAmountValues['fForce']['max'] < amountValuesInSignal[keys]['fForce']:
            minMaxAmountValues['fForce']['max'] = amountValuesInSignal[keys]['fForce']
        if minMaxAmountValues['fForce']['min'] > amountValuesInSignal[keys]['fForce']:
            minMaxAmountValues['fForce']['min'] = amountValuesInSignal[keys]['fForce']

        if minMaxAmountValues['fPosition']['max'] < amountValuesInSignal[keys]['fPosition']:
            minMaxAmountValues['fPosition']['max'] = amountValuesInSignal[keys]['fPosition']
        if minMaxAmountValues['fPosition']['min'] > amountValuesInSignal[keys]['fPosition']:
            minMaxAmountValues['fPosition']['min'] = amountValuesInSignal[keys]['fPosition']

    return minMaxAmountValues


def writeToPickle(path, data):
    with open(path, 'wb') as fileWrite:
        pickle.dump(data, fileWrite)


def readeFromPickle(path):
    with open(path, 'rb') as fileReade:
        data = pickle.load(fileReade)
    return data


# def getMinMaxAmountValueInSignal(signals):
#     result = {}
#
#     minValue = 99999999999
#     maxValue = 0
#     for signal in signals:
#         values = signal['obj']['Measurement']['@fAcceleration'].split(" ")
#
#         if len(values) > maxValue:
#             maxValue = len(values)
#         if len(values) < minValue:
#             minValue = len(values)
#     result['@fAcceleration'] = {'min': minValue, 'max': maxValue}
#
#     minValue = 99999999999
#     maxValue = 0
#     for signal in signals:
#         values = signal['obj']['Measurement']['@fForce'].split(" ")
#
#         if len(values) > maxValue:
#             maxValue = len(values)
#         if len(values) < minValue:
#             minValue = len(values)
#     result['@fForce'] = {'min': minValue, 'max': maxValue}
#
#     return result


def getFileNameFullSignals(filesNames):
    result = []

    for index, fileName in enumerate(filesNames):
        try:
            with open(config.pathToDataXML + "\\" + fileName) as readFile:
                xml = readFile.read()

            doc = xmltodict.parse(xml[3:])
            doc = json.loads(json.dumps(doc))
            if doc['obj']['Measurement']['@fAcceleration'] and doc['obj']['Measurement']['@fForce'] and \
                    doc['obj']['Measurement']['@fPosition']:
                result.append(fileName)
        except:
            pass

    return result
