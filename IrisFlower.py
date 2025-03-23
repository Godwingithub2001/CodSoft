import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("C:\\DATASCIENCE\\MyProjects\\CodSoft\\dataset\\IRIS.csv")
print(df.to_string())

print(df.isnull().sum())
print(df.describe())

x = df.drop('species',axis=1)
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_pred)

with open("IrisFlower_saved.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model saved successfully!")

