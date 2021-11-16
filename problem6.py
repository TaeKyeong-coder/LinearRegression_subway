#파이썬 머신러닝 라이브러리 싸이킷런을 불러오기
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df = pd.read_csv("subway.csv")
df = pd.read_csv("subway.csv", encoding = "cp949")

df.head()


x = df["Year"]
y = df["Line 3"]

#점으로 찍어보기
plt.plot(x,y, marker='o')
plt.show()

line_fitter = LinearRegression()
line_fitter.fit(x.values.reshape(-1,1), y)


line_fitter.predict([[2000]])
line_fitter.predict([[2001]])
line_fitter.predict([[2002]])


