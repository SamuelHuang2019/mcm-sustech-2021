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


class Markov():
    def __init__(self, trans, init=None) -> None:
        self.init = init
        self.trans = trans

    def run(self, times):
        p = self.trans
        for i in range(times - 1):
            p = np.dot(p, self.trans)
        return np.dot(self.init, p)


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
