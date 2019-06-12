import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pprint import pprint as pp

from Sample.Sample import SampleAdapter
from config import config
from filesTools import filesTools


class SOMNetwork(object):
    def __init__(self, input_dim, dim=100, sigma=None, learning_rate=0.01, tay2=1151, dtype=tf.float32):
        if not sigma:
            sigma = dim / 2
        self.dtype = dtype
        # constants
        self.dim = tf.constant(dim, dtype=tf.int64)
        self.learning_rate = tf.constant(learning_rate, dtype=dtype, name='learning_rate')
        self.sigma = tf.constant(sigma, dtype=dtype, name='sigma')
        self.tay1 = tf.constant(1000 / np.log(sigma), dtype=dtype, name='tay1')
        self.minsigma = tf.constant(sigma * np.exp(-1000 / (1000 / np.log(sigma))), dtype=dtype, name='min_sigma')
        self.tay2 = tf.constant(tay2, dtype=dtype, name='tay2')
        # input vector
        self.x = tf.placeholder(shape=[input_dim], dtype=dtype, name='input')
        # iteration number
        self.n = tf.placeholder(dtype=dtype, name='iteration')
        # variables
        self.w = tf.Variable(tf.random_uniform([dim * dim, input_dim], minval=-1, maxval=1, dtype=dtype),
                             dtype=dtype, name='weights')
        # helper
        self.positions = tf.where(tf.fill([dim, dim], True))

    def feed(self, input):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            win_index = sess.run(self.__competition(), feed_dict={self.x: input})
            win_index_2d = np.array(
                [win_index // self.dim.eval(), win_index - win_index // self.dim.eval() * self.dim.eval()])
        return win_index_2d

    def training_op(self):
        win_index = self.__competition('train_')
        with tf.name_scope('cooperation') as scope:
            coop_dist = tf.sqrt(tf.reduce_sum(tf.square(
                tf.cast(self.positions - [win_index // self.dim, win_index - win_index // self.dim * self.dim],
                        dtype=self.dtype)), axis=1))
            sigma = tf.cond(self.n > 1000, lambda: self.minsigma, lambda: self.sigma * tf.exp(-self.n / self.tay1))
            # sigma = tf.cond(self.n > 1000, lambda: self.minsigma, lambda: self.sigma * tf.exp(-1 / self.tay1))
            sigma_summary = tf.summary.scalar('Sigma', sigma)
            tnh = tf.exp(-tf.square(coop_dist) / (2 * tf.square(sigma)))  # topological neighbourhood
        with tf.name_scope('adaptation') as scope:
            lr = self.learning_rate * tf.exp(-self.n / self.tay2)
            minlr = tf.constant(0.01, dtype=self.dtype, name='min_learning_rate')
            lr = tf.cond(lr <= minlr, lambda: minlr, lambda: lr)
            lr_summary = tf.summary.scalar('Learning rate', lr)
            delta = tf.transpose(lr * tnh * tf.transpose(self.x - self.w))
            training_op = tf.assign(self.w, self.w + delta)
        return training_op, lr_summary, sigma_summary

    def __competition(self, info=''):
        with tf.name_scope(info + 'competition') as scope:
            distance = tf.sqrt(tf.reduce_sum(tf.square(self.x - self.w), axis=1))
        return tf.argmin(distance, axis=0)

    def competition(self, info=''):
        return self.__competition(info)


# == Test SOM Network ==

def drawSom(weight, i):
    newWeight = np.zeros([len(weight), len(weight[0]), 3])
    for indexRow, row in enumerate(weight):
        for indexCol, cell in enumerate(row):
            newWeight[indexRow][indexCol] = (np.mean(cell), np.mean(cell), np.mean(cell))

    # minW = newWeight.min()
    # maxW = newWeight.max()
    # pp((minW, maxW))
    newWeight -= np.min(newWeight)
    newWeight /= np.max(newWeight)
    plt.imshow(newWeight)
    plt.savefig(config.pathToData + '\\SOM\\som__' + str(i) + '.png', format='png', dpi=500)
    plt.clf()
    # plt.show()


def test_som_with_color_data(data, dim):
    # test_data = np.random.uniform(0, 1, (1000, 3))
    test_data = data.copy(True)
    test_data = test_data.drop(['marker_debit'], axis=1)
    # test_data = test_data.drop(['marker_well'], axis=1)
    # test_data = test_data.drop(['marker_state'], axis=1)
    values = test_data.values
    values = values - np.min(values)
    values = values / np.max(values) * 2 - 1
    test_data = values
    som_dim = dim
    som = SOMNetwork(input_dim=len(test_data[0]), dim=som_dim, dtype=tf.float64, learning_rate=0.1, tay2=len(data))
    training_op, lr_summary, sigma_summary = som.training_op()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        start = time.time()
        amount = len(test_data)
        ciuntI = 300
        for i, color_data in enumerate(test_data):
            sess.run(training_op, feed_dict={som.x: color_data, som.n: i})
            if i % ciuntI == 0:
                print('iter:{}/{}'.format(i, amount))
                img2 = tf.reshape(som.w, [som_dim, som_dim, -1]).eval()
                drawSom(img2, i)
                end = time.time()
                print(((end - start) / ciuntI) * (amount - i) / 60)
                start = time.time()
        img2 = tf.reshape(som.w, [som_dim, som_dim, -1]).eval()
        drawSom(img2, i)
        img2 = som.w.eval()
        filesTools.saveToPickle(config.pathToPickle + "\\som_debit_norm_{}.pickle".format(dim), img2)
        return img2


if __name__ == "__main__":
    def feed(x):
        # pp(w.shape)
        r = int(np.sqrt(w.shape[0]))
        we = np.linalg.norm(np.sqrt(np.square(w - x)), axis=1)
        index = np.argmin(we)
        i, j = divmod(index, r)
        return i, j


    def norm(drawArray):
        # pp('-' * 150)
        ww = np.copy(drawArray)
        www = np.reshape(ww, (-1, 3))
        amounts = [1, 1, 1]
        for i, c in enumerate(www):
            if c[0] > amounts[0]:
                amounts[0] = c[0]
            if c[1] > amounts[1]:
                amounts[1] = c[1]
            if c[2] > amounts[2]:
                amounts[2] = c[2]
        # pp(np.min(ww))
        pp(amounts)
        for r in ww:
            for c in r:
                if c[0] > 0:
                    c[0] = c[0] / amounts[0] / 2 + 0.5
                if c[1] > 0:
                    c[1] = c[1] / amounts[1] / 2 + 0.5
                if c[2] > 0:
                    c[2] = c[2] / amounts[2] / 2 + 0.5
        # pp(np.min(ww))
        return np.copy(ww)


    dim = 100
    dictDF = filesTools.openFromPickle(config.pathToPickle + "\\dinamos_debit_DICT.pickle")
    debitDict = filesTools.getDebitWell()

    sampleAdapter = SampleAdapter(dictDF=dictDF, debitDict=debitDict)
    # sample = sampleAdapter.getByWell()
    # sample = sampleAdapter.getByWellNorm()
    sample = sampleAdapter.getByDebitNorm()
    # sample = sampleAdapter.getByParts(2)

    # pp(sample['marker_debit'].value_counts())
    # pp(sample['marker_well'].value_counts())
    # pp(sample['marker_state'].value_counts())

    # w = test_som_with_color_data(sample.sample(len(sample)), dim)
    # sample = sample.sample(len(sample))
    # sample = sample.reset_index(drop=True)
    w = filesTools.openFromPickle(config.pathToPickle + "\\som_debit_norm_{}.pickle".format(dim))

    # colors = {
    #     'marker_well': {
    #         'Скважина 15795': (1, 0, 0),
    #         'Скважина 18073': (0, 1, 0),
    #         'Скважина 30065': (0, 0, 1),
    #     },
    #     'marker_state': {
    #         'bad': (1, 0, 0),
    #         'good': (0, 1, 0),
    #     }
    # }
    marker = sample[['marker_debit']]

    colors = marker - np.min(marker)
    colors = colors / np.max(colors)
    test_data = sample.copy(True)
    test_data = test_data.drop(['marker_debit'], axis=1)
    # test_data = test_data.drop(['marker_state'], axis=1)
    values = test_data.values
    values = values - np.min(values)
    values = values / np.max(values) * 2 - 1
    sample = values
    drawArray = np.full((dim, dim, 3), 0.)
    amount = len(sample)
    for index, x in enumerate(sample[::]):
        i, j = feed(x)
        drawArray[i, j] = (colors.loc[index], 0, 0)
        if index % 100 == 0:
            pp('Пройдено {} из {}'.format(index, amount))
            # pp(drawArray)
            # drawArray -= np.min(drawArray)
            # pp(drawArray)
            # drawArray /= np.max(drawArray)
            # for i, row in enumerate(drawArray):
            #     for j, cell in enumerate(row):
            #         plt.scatter(i, j, s=10, c=(cell, cell, cell), marker="o")
            #     pp(i)
            # drawArray = drawArray / np.max(drawArray)
            # newDrawArray = norm(drawArray)
            plt.imshow(drawArray)
            plt.savefig(config.pathToData + '\\som_well_norm_{}.png'.format(dim), format='png', dpi=500)
            plt.clf()
    newDrawArray = norm(drawArray)
    plt.imshow(newDrawArray)
    plt.savefig(config.pathToData + '\\som_well_norm_{}.png'.format(dim), format='png', dpi=500)
    plt.clf()
