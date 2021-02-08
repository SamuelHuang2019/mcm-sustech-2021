import pandas as pd
from matplotlib import pyplot as plt

file = '../data/fungi_data.csv'
df = pd.read_csv(file)

plt.grid()
plt.scatter(df['width'], df['rate'], s=df['d-122'] * 20, cmap='plasma', alpha=0.8, edgecolors='grey')
plt.xlabel('Moisture niche width / $MPa$')
plt.ylabel('Hyphal extension rate / $mm\cdot day^{-1}$')
plt.title('Hyphal extension rate and moisture niche width of fungi')
plt.show()
