import copy
from pprint import pprint as pp

import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import random

from sklearn.cluster import KMeans

import plotly.graph_objs as go
import plotly


def drawDinamo(dinamo: pd.DataFrame):
    """
        Функция отрисовывает днамограмму

        Параметры:
            dinamo: значения динамограммы
    """
    array = dinamo.values
    plt.plot(array[1::2], array[0:-1:2], 'b')
    plt.show()


def TNSE(data: pd.DataFrame, marker="marker_debit"):
    """
        Функия преобразует меногомерные данные в двумерные
    
        Параметры: 
            data - данные 
            marker - изучаемая характеристика
            
        Возвращает:
            pandas.DataFrame
    """
    tsne = TSNE()
    markers = data[[marker]].copy(True)
    inputData = data.drop([marker], axis=1)
    X_tsne = tsne.fit_transform(inputData)
    result = pd.DataFrame(X_tsne)
    result[marker] = markers
    return result


def drawSemple(sample: pd.DataFrame, path='semple.html', marker_in="marker_well"):
    """
        Функция отображает на графике полученные данные
        Параметры:
            samole    - данные для отрисовки
            path      - путь, по которому сохранить изображение
            marker_in - изображаемая характеристика
    """
    markers = sample[marker_in].unique()
    sampleCP = sample.copy(True)
    sampleCP[marker_in] = sampleCP[marker_in] - np.min(sampleCP[marker_in])
    sampleCP[marker_in] = sampleCP[marker_in] / np.max(sampleCP[marker_in])
    colors = markers - np.min(markers)
    colors = colors / np.max(colors) * 255
    colors = np.array(colors, dtype=int)
    traces = []
    pp(markers)
    for index, marker in enumerate(markers):
        random.seed(index * 10)
        traces.append(
            go.Scatter(
                x=sample.loc[sample[marker_in] == marker][0],
                y=sample.loc[sample[marker_in] == marker][1],
                name=str(marker),
                mode='markers',
                marker=dict(
                    size=10,
                    color='rgb(' + ",".join(
                        [str(random.randint(0, 255)), str(random.randint(0, 255)), str(random.randint(0, 255))]) + ')',
                    # [str(colors[index]), str(0), str(0)]) + ')',
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


def analizeFeatureInput(featureInput, sample):
    array = np.array(featureInput)
    tmp = copy.deepcopy(featureInput)
    result = []
    for i in range(1):
        result.append(tmp.pop(np.argmax(tmp)))

    indexes = []
    for item in result:
        indexes.append(np.where(array == item)[0][0])

    indexes = sorted(indexes)
    pp(indexes)
    plt.plot(result, 'b')
    plt.show()

    positionX = []
    positionY = []

    for index in indexes:
        if index % 2 == 0:
            if index not in positionX:
                positionX.append(index)
            if index + 1 not in positionY:
                positionY.append(index + 1)
        else:
            if index - 1 not in positionX:
                positionX.append(index - 1)
            if index not in positionY:
                positionY.append(index)
    pp(len(positionX))
    pp(len(positionY))

    sample_loc = sample.copy(True)

    marker = sample_loc.pop("marker_debit")

    plt.plot(sample_loc.mean().values[1::2], sample_loc.mean().values[0::2], "b")
    plt.plot(sample_loc.mean().values[positionY], sample_loc.mean().values[positionX], "ro")
    plt.show()


def k_means(data: pd.DataFrame, k: int):
    test_data = data

    test_data.pop('marker_well')
    test_data.pop('marker_state')

    model = KMeans(n_clusters=k, n_jobs=-1)
    model.fit(test_data)
    all_predictions = model.predict(test_data)
    test_data['class'] = all_predictions

    return test_data


def elbow_method_k_means(data, end: int, start: int = 1, step: int = 1):
    X = []
    Y = []
    for k in range(start, end, step):
        model = KMeans(n_clusters=k, n_jobs=-1)
        model.fit(data)
        X.append(k)
        Y.append(model.inertia_)
    plt.plot(X, Y, 'b')
    plt.show()


def tree(data: pd.DataFrame, k=10):
    # Создаем датафрейм
    seeds_df = data

    # Исключаем информацию об образцах зерна, сохраняем для дальнейшего использования
    varieties = list(seeds_df.pop('marker_state'))
    varieties = list(seeds_df.pop('marker_well'))

    # Извлекаем измерения как массив NumPy
    samples = seeds_df.values

    # Реализация иерархической кластеризации при помощи функции linkage
    mergings = linkage(samples, method='average')

    # pp(mergings)

    # dendrogram(mergings, p=100, truncate_mode='lastp')
    # plt.plot(mergings.transpose()[1])
    # plt.show()

    elbow = []

    groups = fcluster(mergings, k, criterion='maxclust')

    seeds_df['class'] = groups
    return seeds_df


def wgss(data, groups):
    _data = np.array(data)
    res = 0.0
    for cluster in groups:
        inclust = _data[np.array(groups) == cluster]
        meanval = np.mean(inclust, axis=0)
        res += np.sum((inclust - meanval) ** 2)
        return res
