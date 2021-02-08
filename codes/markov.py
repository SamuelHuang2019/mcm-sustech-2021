# %%
import numpy as np
from matplotlib import pyplot as plt
from softmax import softmax


def cal_trans(r) -> np.array:
    # s = sum(log(e) for e in r)
    # return np.array([[log(e) / s for e in r]] * len(r))
    trans = np.zeros([len(r)] * 2)
    for i in range(len(r)):
        # row = [e / r[i] for e in r]
        row = [- r[i] / e for e in r]
        trans[i] = softmax(np.array(row))
    return trans


def renv(r, d, w: int):
    ls = []
    for i in range(len(r)):
        ls.append(r[i] * d[i] / (w + d[i]))
    return ls


# def decomposition_d(r):
#     init = np.array([100] * len(r))
#     markov = Markov(cal_trans(r), init)
#     dist = markov.run(100)
#     return dist,


class Markov():
    def __init__(self, trans, init=None) -> None:
        self.init = init
        self.trans = trans

    def run(self, times):
        p = self.trans
        for i in range(times - 1):
            # p = np.dot(p, self.trans)
            p = p @ self.trans
        # return np.dot(self.init, p)
        return self.init @ p


def dist(r):
    init = np.array([100] * len(r))
    markov = Markov(cal_trans(r), init)
    dist = markov.run(100)
    return dist / (100 * len(r))


def decomposition_r(r):
    # return np.polyval([1.6850, 3.4082], r)
    return 1.6850 * r + 3.482


def decomposition_m(m):
    return np.exp(0.5831 * m + 0.4896)


def decomposition(r, dist, time):
    wr = r @ dist

    def d(k, b):
        return k * wr + b

    if time == 3:
        return d(1.539, 17.31)
    if time == 5:
        return d(1.237, 57.87)
    if time == 122:
        return decomposition_r(wr)


# %%
r = np.array([8.75, 10.8, 0.49, 1.07])
# r = np.array([8.75, 10.8, 10, 20])
# r = np.array([1, 1, 1, 1])
init = np.array([100] * 4)
trans = cal_trans(r)
print(trans)
markov = Markov(trans, init)

ls = [init]
for i in range(8):
    ls.append(markov.run(i))

result = np.array(ls)
result = result / 4
# y = make_interp_spline(range(20), result)

plt.plot(result)
plt.legend(('1', '2', '3', '4'), loc='best')
plt.xlabel('iteration')
plt.ylabel('relative biomass / percent')
plt.title('Predicted fungi community composition')
plt.show()

# plt.figure()
# plt.pie(result[-1], labels=[1, 2, 3, 4])

# %% Bio-div 4 species, no env
r4 = np.array([8.75, 10.8, 8.51, 9.62])
dist4 = dist(r4)
d122_4 = decomposition(r4, dist4, 122)
d3_4 = decomposition(r4, dist4, 3)
d5_4 = decomposition(r4, dist4, 5)

# %% Bio-div 6 species, no env
r6 = np.append(r, [[3.88, 10.62]])
dist6 = dist(r6)
d122_6 = decomposition(r6, dist6, 122)
d3_6 = decomposition(r6, dist6, 3)
d5_6 = decomposition(r6, dist6, 5)

# x = np.arange(3)
y4 = np.array([d122_4, d3_4, d5_4])
y6 = np.array([d122_6, d3_6, d5_6])
# y6 = y6 - y4 + 1
# y4 = [1] * 3
tick_label = ['122 days', '3 years', '5 years']
bar_width = 0.2

x = np.arange(1)
plt.subplot(311)
plt.barh(x, y4[0], bar_width, label='4-species')
plt.barh(x + bar_width, y6[0], bar_width, label='6-species')
plt.yticks(x + bar_width / 2, ['122 days'])
plt.xlim(xmin=d122_4 - 1, xmax=d122_6 + 20)
plt.legend()

plt.subplot(312)
plt.barh(x, y4[1], bar_width)
plt.barh(x + bar_width, y6[1], bar_width)
plt.yticks(x + bar_width / 2, ['3 years'])
plt.xlim(xmin=d3_4 - 1, xmax=d3_6 + 2)

plt.subplot(313)
plt.barh(x, y4[2], bar_width)
plt.barh(x + bar_width, y6[2], bar_width)
plt.yticks(x + bar_width / 2, ['5 years'])
plt.xlim(xmin=d5_4 - 1, xmax=d5_6 + 0.5)

plt.show()
