import numpy as np
from pprint import pprint as pp
from scipy.spatial.distance import euclidean
import plotly.plotly as py
import plotly.graph_objs as go
from time import time

from Class.Sample import SampleAdapter
from config import config
import plotly

from developTools import toolsFile


class SOM(object):
    def __init__(self, input_size):
        self.input_size = input_size
        self.sample = []
        self.weights = []
        self.train = []
        self.test = []
        self.height = input_size
        self.width = input_size

    def init_w(self):
        pass


if __name__ == "__main__":
    dim = 100
    sigma0 = 1
    learn0 = 0.01
    tau1 = 1000
    tau2 = 10000

    dictDF = toolsFile.openFromPickle(config.pathToPickle + "\\dinamos_DICT.pickle")
    sampleAdapter = SampleAdapter(dictDF=dictDF)
    sample = sampleAdapter.getByWell()
    sample = sample.drop(['marker'], axis=1)
    sample = sample.values
    #
    # sample = np.random.uniform(0, 1, (5000, 10))
    #
    weight = np.random.uniform(np.min(sample), np.max(sample), (dim, dim, len(sample[0])))
    resI = 0
    resJ = 0


    def findWin(x):
        minDist = 99999999999
        resI, resJ = 0, 0
        for i, row in enumerate(weight):
            for j, cell in enumerate(row):
                dist = euclidean(cell, x)
                if minDist > dist:
                    minDist = dist
                    resI, resJ = i, j
        return resI, resJ


    def sigma(t):
        s = sigma0 * np.exp(-t / tau1)
        # pp("sigma: {}".format(s))
        return s


    def h(t, win):
        top = np.zeros((dim, dim))
        s = sigma(t)
        for i, row in enumerate(top):
            for j, cell in enumerate(row):
                dist = euclidean((i, j), win)
                top[i, j] = np.exp(-(dist ** 2) / (2 * s ** 2))
        # pp(top)
        # draw(top)
        return top


    def learn(t):
        l = learn0 * np.exp(-t / tau2)
        # pp("learn: {}".format(l))
        return l


    def draw(weight):
        traces = []
        weight = weight - np.min(weight)
        weight = weight / np.max(weight)
        for i, row in enumerate(weight):
            for j, cell in enumerate(row):
                traces.append(go.Scatter(
                    x=[i],
                    y=[j],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='rgb(' + ",".join(
                            (str(round(np.linalg.norm(cell) * 1, 3)), str(round(np.linalg.norm(cell) * 1, 3)),
                             str(round(np.linalg.norm(cell) * 1, 3)))) + ')',
                    )
                )
                )
        layout = dict(title='Динамограммы',
                      yaxis=dict(zeroline=False),
                      xaxis=dict(zeroline=False)
                      )
        fig = dict(data=traces, layout=layout)
        plotly.offline.plot(fig, filename=config.pathToData + "\\som.html")


    amountEx = len(sample)
    for t, x in enumerate(sample):
        if t % 100 == 0:
            start = time()
        win = findWin(x)
        error = np.dot(learn(t), np.dot(h(t, win), (weight - x)))
        weight += error
        if t % 100 == 0:
            inerval = time() - start
            pp("Примеров пройдено {} из {} для обучения".format(t + 1, amountEx))
            pp(inerval * (amountEx - t) / 60)
    pp("Конец обучения")

    toolsFile.saveToPickle(config.pathToPickle + "\\som_weight.pickle", weight)
    # weight = toolsFile.openFromPickle(config.pathToPickle + "\\som_weight.pickle")
    drawArray = np.zeros((dim, dim))
    start = 0
    for t, x in enumerate(sample):
        start += time()
        i, j = findWin(x)
        drawArray[i, j] += 1
        if t % 200 == 0:
            inerval = time() - start / 200
            start = 0
            pp("Примеров пройдено {} из {} для отрисовки".format(t + 1, amountEx))
            pp(inerval * (amountEx - t) / 60)
    pp("Конец подготовки создания массива для отрисовки")
    drawArray -= np.min(drawArray)
    drawArray /= np.max(drawArray)
    toolsFile.saveToPickle(config.pathToPickle + "\\som_result.pickle", drawArray)
    draw(drawArray)
