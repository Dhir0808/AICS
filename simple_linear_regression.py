import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

df = pd.read_csv('D:\\Bluetooth\\Salary_Data.csv')

plt.title('Linear Regression')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.scatter(df.YearsExperience, df.Salary, color='red', marker='+')

reg = linear_model.LinearRegression()
reg.fit(df[['YearsExperience']], df.Salary)

plt.plot(df.YearsExperience, reg.predict(df[['YearsExperience']]), color='blue')
print("Slope: ", reg.coef_)
print("Intercept: ", reg.intercept_)
plt.show()