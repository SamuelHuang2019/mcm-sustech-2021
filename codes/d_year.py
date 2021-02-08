import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('..\data\decomposition_y.CSV')
y3 = df[df['years'] == 3]
y5 = df[df['years'] == 5]

# plt.scatter(y3['extension rate'], y3['mass loss'])
# plt.scatter(y5['extension rate'], y5['mass loss'])
# plt.show()


g = sns.lmplot(data=df, x="extension rate", y="mass loss", hue='years')
# g.fig.set_figwidth(5.58)
g.fig.set_figheight(5 * 1.2)
sns.set_context('talk')

plt.xlabel('community-weighted hyphal extension rate / $mm\cdot day^{-1}$')
plt.ylabel('mass loss / %')
plt.show()

z3 = np.polyfit(y3['extension rate'], y3['mass loss'], 1)
p3 = np.poly1d(z3)
z5 = np.polyfit(y5['extension rate'], y5['mass loss'], 1)
p5 = np.poly1d(z5)
