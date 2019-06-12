import copy
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
    markers = pd.DataFrame()
    if 'marker_well' in data.columns:
        markers['marker_well'] = data.pop('marker_well')
    if 'marker_state' in data.columns:
        markers['marker_state'] = data.pop('marker_state')
    if 'marker_debit' in data.columns:
        markers['marker_debit'] = data.pop('marker_debit')
    if 'class' in data.columns:
        markers['class'] = data.pop('class')

    tsne = TSNE()
    X_tsne = tsne.fit_transform(data)
    result = pd.DataFrame(X_tsne)
    result[markers.columns] = markers
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
    for index, marker in enumerate(markers):
        if marker_in == 'marker_debit':
            color = [str(colors[index]), str(0), str(0)]
        else:
            color = [str(random.randint(0, 255)), str(random.randint(0, 255)), str(random.randint(0, 255))]

        random.seed(index * 10)
        traces.append(
            go.Scatter(
                x=sample.loc[sample[marker_in] == marker][0],
                y=sample.loc[sample[marker_in] == marker][1],
                name=str(marker),
                mode='markers',
                marker=dict(
                    size=10,
                    color='rgb(' + ",".join(color) + ')',
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

    sample_loc = sample.copy(True)

    markers = pd.DataFrame()
    if 'marker_well' in sample_loc.columns:
        markers['marker_well'] = sample_loc.pop('marker_well')
    if 'marker_state' in sample_loc.columns:
        markers['marker_state'] = sample_loc.pop('marker_state')
    if 'marker_debit' in sample_loc.columns:
        markers['marker_debit'] = sample_loc.pop('marker_debit')
    if 'class' in sample_loc.columns:
        markers['class'] = sample_loc.pop('class')

    plt.plot(sample_loc.mean().values[1::2], sample_loc.mean().values[0::2], "b")
    plt.plot(sample_loc.mean().values[positionY], sample_loc.mean().values[positionX], "ro")
    plt.show()


def k_means(data: pd.DataFrame, k: int):
    test_data = data

    markers = pd.DataFrame()
    if 'marker_well' in data.columns:
        markers['marker_well'] = data.pop('marker_well')
    if 'marker_state' in data.columns:
        markers['marker_state'] = data.pop('marker_state')
    if 'marker_debit' in data.columns:
        markers['marker_debit'] = data.pop('marker_debit')
    if 'class' in data.columns:
        markers['class'] = data.pop('class')

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
    seeds_df = data
    markers = pd.DataFrame()
    if 'marker_well' in seeds_df.columns:
        markers['marker_well'] = seeds_df.pop('marker_well')
    if 'marker_state' in seeds_df.columns:
        markers['marker_state'] = data.pop('marker_state')
    if 'marker_debit' in seeds_df.columns:
        markers['marker_debit'] = seeds_df.pop('marker_debit')
    if 'class' in seeds_df.columns:
        markers['class'] = seeds_df.pop('class')

    samples = seeds_df.values
    mergings = linkage(samples, method='average')
    groups = fcluster(mergings, k, criterion='maxclust')
    seeds_df['class'] = groups
    return seeds_df
