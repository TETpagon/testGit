import os
from pprint import pprint as pp

import pandas as pd

from config import config

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


def getSample():
    dataFromStorage = getDataFromFiles()
    return dataFromStorage


def getDataFromFiles():
    data = []

    for dir in os.listdir(config.pathToDirOilsWell):
        pathToDate = config.pathToDirOilsWell + "\\" + dir
        for date in os.listdir(pathToDate):
            pathToFiles = pathToDate + "\\" + date
            for filename in os.listdir(pathToFiles):
                pathToFile = pathToFiles + "\\" + filename
                df = pd.read_csv(pathToFile, sep=";")
                if df.dtypes["force" and "position"] != 'object':
                    dfs = preprocessData(df)
                    data += dfs
                # break
            # break
        # break
    return data


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


def drawDinamo(dinamo):
    plt.plot(dinamo["position"], dinamo["force"], 'b')
    plt.show()


def convertDFTOArray(listDF):
    newList = []

    for df in listDF:
        dfNew = df.copy(True)
        dfNew = dfNew.sample(577)
        dfNew.sort_index()
        array = dfNew.values
        array = array.ravel()
        newList.append(array)
    return np.array(newList)


def convertDFTOArray2D(listDF):
    newList = []

    for df in listDF:
        dfNew = df.copy(True)
        # dfNew = dfNew.sample(577)
        dfNew = dfNew.loc[-577:]
        dfNew.sort_index()
        array = dfNew.values
        # array = array.ravel()
        newList += list(array)

    return np.array(newList)


def TNSE(data):
    tsne = TSNE()

    X_tsne = tsne.fit_transform(data)
    X_tsne = X_tsne.transpose()

    plt.plot(X_tsne[0], X_tsne[1], 'bo')
    plt.savefig(config.pathToData + '\\TNSE.png', format='png', dpi=500)
    plt.clf()
    # plt.show()


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
