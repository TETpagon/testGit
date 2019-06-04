import os
from pprint import pprint as pp

import pandas as pd
from datetime import datetime

from config import config
from developTools import createSemple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white')
# %matplotlib inline
from sklearn import decomposition
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.manifold import TSNE

import random

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from sklearn.cluster import dbscan

import plotly.plotly as py
import plotly.graph_objs as go
import plotly


def getSample():
    dataFromStorage = getDataFromFiles()
    return dataFromStorage


def getDebitByDateTime(well_oil, well, date, time):
    pp((well, date, time))
    dateDinamo = datetime.strptime(date, "%d-%m-%y")
    timeDinamo = datetime.strptime(time, "%H-%M")
    # dateDebit = datetime.strptime()
    ckeckDate = (str(dateDinamo.date().day) if len(str(dateDinamo.date().day)) > 2 else "0" + str(
        dateDinamo.date().day)) + "." + (
                    str(dateDinamo.date().month) if len(str(dateDinamo.date().month)) > 2 else "0" + str(
                        dateDinamo.date().month)) + "." + str(dateDinamo.year)[2:]
    dfDate = well_oil[well][well_oil[well]['date'] == ckeckDate]


def getDataFromFiles():
    dataList = []
    dataDict = {}
    # well_oil = createSemple.getDebit()
    for dir in os.listdir(config.pathToDirOilsWell):
        pathToDate = config.pathToDirOilsWell + "\\" + dir
        dataDict[dir] = {}
        for date in os.listdir(pathToDate):
            pathToFiles = pathToDate + "\\" + date
            dataDict[dir][date] = {}
            for filename in os.listdir(pathToFiles):
                pathToFile = pathToFiles + "\\" + filename
                df = pd.read_csv(pathToFile, sep=";")
                if df.dtypes["force" and "position"] != 'object':
                    # getDebitByDateTime(well_oil, dir, date, filename[:-4])
                    dfs = preprocessData(df)
                    dataDict[dir][date][filename[:-4]] = dfs
                    dataList += dfs
                    break
            break
        break
    return dataList, dataDict


def preprocessData(df: pd.DataFrame):
    data = df.loc[df["force" and "position"].notnull()][["index", "force", "position"]]
    data = data.reset_index()[["index", "force", "position"]]
    indexes = list(data[data["index"].notnull()].index)
    indexes.append(data.index[-1])
    dinamos = []
    start = indexes[0]
    for index in indexes[1:]:
        dinamos.append(data[["force", "position"]].loc[start: index - 1].reset_index()[["force", "position"]])
        start = index
    return dinamos


def PCA(listDF: list):
    pca = decomposition.PCA(n_components=2)
    X_reduced = pca.fit_transform(listDF)
    X_reduced = X_reduced.transpose()

    # И нарисуем получившиеся точки в нашем новом пространстве
    plt.plot(X_reduced[1], X_reduced[0], 'bo')
    plt.savefig(config.pathToData + '\\PCA.png', format='png', dpi=500)
    plt.clf()
    # plt.show()


def drawDinamo(dinamo:pd.DataFrame):
    array = dinamo.values
    plt.plot(array[1::2], array[0:-1:2], 'b')
    plt.show()

#def drawDinamo(dinamo):
#     plt.plot(dinamo["position"], dinamo["force"], 'b')
#     plt.show()


def convertDFTOArray(listDF, part_in=9):
    newList = []
    if 576 % part_in == 0:
        part = part_in
    else:
        part = 9
    pp("part: {}".format(part))

    for df in listDF:
        dfNew = df.copy(True)
        start = len(dfNew) - 576
        dfNew = dfNew.loc[start:]
        dfNew.sort_index()
        array = dfNew.values
        array = array.ravel()
        newarray = [{'part': part, 'values': item} for part, item in enumerate(np.split(array, part))]
        newList += list(newarray)
        # break
    return np.array(newList)


def TNSE(data: pd.DataFrame):
    # tsne = TSNE()
    # markers = data[['marker_well', 'marker_state']].copy(True)
    # inputData = data.drop(['marker_well'], axis=1)
    # inputData = inputData.drop(['marker_state'], axis=1)
    # X_tsne = tsne.fit_transform(inputData)
    # result = pd.DataFrame(X_tsne)
    # result[['marker_well', 'marker_state']] = markers
    # return result
    tsne = TSNE()
    markers = data[['marker_debit']].copy(True)
    inputData = data.drop(['marker_debit'], axis=1)
    X_tsne = tsne.fit_transform(inputData)
    result = pd.DataFrame(X_tsne)
    result[['marker_debit']] = markers
    return result


def DBSCAN(listDF):
    clustering, labels = dbscan(listDF, eps=10, min_samples=10)
    tmp = []

    for item in labels:
        tmp.append(item)

    df = pd.Series(tmp)
    pp(df.unique())
    # pp(len(labels))


def qwe(listDF):
    tmp = []

    for item in listDF:
        tmp.append(len(item))

    df = pd.DataFrame(tmp)
    pp(df.describe())


def spectr(listDF):
    # X = np.zeros((150, 2))
    #
    # np.random.seed(seed=42)
    # X[:50, 0] = np.random.normal(loc=0.0, scale=.3, size=50)
    # X[:50, 1] = np.random.normal(loc=0.0, scale=.3, size=50)
    #
    # X[50:100, 0] = np.random.normal(loc=2.0, scale=.5, size=50)
    # X[50:100, 1] = np.random.normal(loc=-1.0, scale=.2, size=50)
    #
    # X[100:150, 0] = np.random.normal(loc=-1.0, scale=.2, size=50)
    # X[100:150, 1] = np.random.normal(loc=2.0, scale=.5, size=50)

    distance_mat = pdist(listDF[:500])  # pdist посчитает нам верхний треугольник матрицы попарных расстояний

    Z = hierarchy.linkage(distance_mat, 'single')  # linkage — реализация агломеративного алгоритма
    pp(len(Z))
    plt.figure(figsize=(10, 5))
    dn = hierarchy.dendrogram(Z, color_threshold=0.5)
    plt.show()


def andrews(listDF):
    def andrews_curve(x, theta):
        curve = list()
        for th in theta:
            x1 = x[0] / np.sqrt(2)
            x2 = x[1] * np.sin(th)
            x3 = x[2] * np.cos(th)
            x4 = x[3] * np.sin(2. * th)
            curve.append(x1 + x2 + x3 + x4)
        return curve

    accuracy = 1000
    samples = listDF
    theta = np.linspace(-np.pi, np.pi, accuracy)

    # for s in samples[:595]:  # setosa
    #     plt.plot(theta, andrews_curve(s, theta), 'r')
    #
    # for s in samples[595:1184]:  # versicolor
    #     plt.plot(theta, andrews_curve(s, theta), 'g')

    for s in samples[:10]:  # virginica
        plt.plot(theta, andrews_curve(s, theta), 'b')

    plt.xlim(-np.pi, np.pi)
    plt.show()


def test():
    df = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # pp(df)
    df.copy(True)
    df = df.sample(5)
    pp(df.sort_index())


def drawSemple(sample: pd.DataFrame, path='semple.html', marker_in="marker_well"):
    markers = sample[marker_in].unique()
    sampleCP = sample.copy(True)
    sampleCP[marker_in] = sampleCP[marker_in] - np.min(sampleCP[marker_in])
    sampleCP[marker_in] = sampleCP[marker_in] / np.max(sampleCP[marker_in])
    colors = markers - np.min(markers)
    colors = colors / np.max(colors) * 255
    colors = np.array(colors, dtype=int)
    traces = []
    for index, marker in enumerate(markers):
        traces.append(
            go.Scatter(
                x=sample.loc[sample[marker_in] == marker][0],
                y=sample.loc[sample[marker_in] == marker][1],
                name=str(marker),
                mode='markers',
                marker=dict(
                    size=10,
                    color='rgb(' + ",".join(
                        # [str(random.randint(0, 255)), str(random.randint(0, 255)), str(random.randint(0, 255))]) + ')',
                        [str(colors[index]), str(0), str(0)]) + ')',
                    line=dict(
                        width=2,
                        color='rgb(0, 0, 0)'
                    )
                )
            )
        )

    layout = dict(title='Динамограммы',
                  yaxis=dict(zeroline=False),
                  xaxis=dict(zeroline=False)
                  )
    fig = dict(data=traces, layout=layout)
    plotly.offline.plot(fig, filename=path)
