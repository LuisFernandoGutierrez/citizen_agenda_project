import pandas as pd


file_name = 'C:/Users/gabriel/localhost/medellin_2020_12/tweets4.xlsx'
df = pd.read_excel(file_name)
print(df.describe())
print(df.head(10))
