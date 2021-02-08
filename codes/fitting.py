import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

sns.set(font_scale=1.5, style="white")

file = '../data/fungi_data.csv'
df = pd.read_csv(file)

x = df['rate']
x = np.array(x)
y = df['d-122']
y = np.array(y)


def func1(x, a, b):
    return a * x + b


k, b = curve_fit(func1, x, y)[0]
print(k, b)

g = sns.lmplot(data=df, x="rate", y="d-122", aspect=1.2)
g.fig.set_figwidth(6.8 * 1.1)
g.fig.set_figheight(5.2 * 1.1)
plt.xlabel('hyphal extension rate / $mm·day^{-1}$')
plt.ylabel('decomposition rate / %')
plt.show()

df['new'] = np.log(df['d-122'])
x = df['tolerance']
x = np.array(x)
y = df['new']
y = np.array(y)


def func2(x, a, b):
    return np.exp(a * x + b)


k, b = curve_fit(func2, x, y)[0]
print(k, b)

g = sns.lmplot(data=df, x="tolerance", y="new", aspect=1.2)
g.fig.set_figwidth(6.8 * 1.1)
g.fig.set_figheight(5.2 * 1.1)
plt.xlabel('moisture tolerance')
plt.ylabel('$\ln$(decomposition rate)')
plt.show()
