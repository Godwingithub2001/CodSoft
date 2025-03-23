import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:\\DATASCIENCE\\MyProjects\\CodSoft\\dataset\\advertising.csv")

print(df.describe())

print(df.isnull().sum())

x = df[['TV','Radio','Newspaper']]
y = df[['Sales']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_pred)



with open("Sales.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model saved successfully!")