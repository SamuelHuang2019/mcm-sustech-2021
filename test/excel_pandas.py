import pandas as pd

df = pd.read_excel(r'C:\Users\Guanc\Desktop\test.xlsx', sheet_name='Sheet1')

start = 0

for i in range(len(df)):
    if df.loc[i, 'B'] != df.loc[start, 'B']:
        df.loc[start, 'D'] = sum(df.loc[start:i + 1, 'C'])
        start = i + 1

df.to_excel(r'C:\Users\Guanc\Desktop\test.xlsx', sheet_name='Sheet1')
