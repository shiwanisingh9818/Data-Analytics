# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:14:35 2020

@author: shiwa
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston
dataset = load_boston()

X = dataset.data
y = dataset.target

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

r2 = lin_reg.score(X, y)
r2

m = X.shape[0]
k = X.shape[1]

adj_r2 = 1 - (((1 - r2) * (m - 1)) / (m - k - 1))
adj_r2

r-squared = 0.7406
adj r-squared = 0.7337

#################### Adding Impurities #####################

irrelevant_col = np.random.randn(506, 3)

X_new = np.c_[X, irrelevant_col]
X_new.shape

from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X_new, y)

r2 = lin_reg_1.score(X_new, y)
r2

m = X_new.shape[0]
k = X_new.shape[1]

adj_r2 = 1 - (((1 - r2) * (m - 1)) / (m - k - 1))
adj_r2

r-squared = 0.7416
adj r-squared = 0.7331
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)
y = np.dot(X, beta) + e

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


########################  GRE Admission Dataset #####################

dataset = pd.read_csv('Admission_Predict_Ver1.1.csv')
dataset.isnull().sum()

desc = dataset.describe()

plt.hist(dataset['GRE Score'], bins = 100)
plt.show()

plt.hist(dataset['TOEFL Score'], bins = 100)
plt.show()

plt.hist(dataset['University Rating'], bins = 10)
plt.show()

plt.hist(dataset['SOP'], bins = 100)
plt.show()

plt.hist(dataset['LOR '], bins = 100)
plt.show()

plt.hist(dataset['CGPA'], bins = 100)
plt.show()

plt.hist(dataset['Research'], bins = 10)
plt.show()

plt.hist(dataset['Chance of Admit '], bins = 100)
plt.show()


pd.plotting.scatter_matrix(dataset)

corr_mat = dataset.corr()

import seaborn as sns
sns.heatmap(corr_mat, annot = True)

X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.score(X, y)