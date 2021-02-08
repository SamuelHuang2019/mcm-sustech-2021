import numpy as np


def cal_trans(r) -> np.array:
    trans = np.zeros([len(r)] * 2)
    for i in range(len(r)):
        row = [- r[i] / e for e in r]
        trans[i] = softmax(np.array(row))
    return trans


class Markov():
    def __init__(self, trans, init=None) -> None:
        self.init = init
        self.trans = trans

    def run(self, times):
        p = self.trans
        for i in range(times - 1):
            p = p @ self.trans
        return self.init @ p


def dist(r):
    init = np.array([100] * len(r))
    markov = Markov(cal_trans(r), init)
    return markov.run(100) / (100 * len(r))
