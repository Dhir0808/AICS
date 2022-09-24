import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("D:\\Bluetooth\\50_Startups.csv")

reg = linear_model.LinearRegression()
reg.fit(df[['R&D Spend', 'Administration', 'Marketing Spend']], df.Profit)

print(reg.coef_)
print(reg.intercept_)

print(reg.predict([[100000, 99999, 5000000]]))
