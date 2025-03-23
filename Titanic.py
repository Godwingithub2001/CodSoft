# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# loading dataset

data_value = pd.read_csv("C:\\DATASCIENCE\\MyProjects\\CodSoft\\dataset\\Titanic-Dataset.csv")
df = pd.DataFrame(data_value)
print(df.to_string())

# data exploration and cleaning like check for missing values, drop unnecessary columns, fill missing values

data_value.isnull().sum()

data_value.drop(['PassengerId', 'Name','Ticket', 'Cabin'], axis=1, inplace=True)

data_value['Age'].fillna(data_value['Age'].median(), inplace=True)

data_value['Embarked'].fillna(data_value['Embarked'].mode()[0],inplace=True)

# data preprocessing

le = LabelEncoder()
data_value['Sex'] = le.fit_transform(data_value['Sex']) # male:1,female:0
data_value['Embarked'] = le.fit_transform(data_value['Embarked'])


x = data_value.drop('Survived', axis=1)
y = data_value['Survived']



# splitting  dataset into train and test

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# standardize

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train,y_train)

# predictions

y_pred = model.predict(x_test)
print({"Prediction": y_pred})



# Count number of people who survived and who didn't, 0: who did not survive, 1 who survuve

survival_count = data_value['Survived'].value_counts()

print(f"Number of people who did not survive: {survival_count[0]}")
print(f"Number of people who survived: {survival_count[1]}")



# Save the model as a pkl file
with open("Titanic_saved.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model saved successfully!")

