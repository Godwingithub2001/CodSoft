# Import  libraries

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# load dataset and exploring

data_path = pd.read_csv("C:\\DATASCIENCE\\MyProjects\\CodSoft\\dataset\\IMDb Movies India.csv", encoding='ISO-8859-1')
df = pd.DataFrame(data_path)
print(df.to_string())

print(df.head())

print(df.info())

# data preprocessing and feature engineering

# Drop rows with missing ratings
df = df.dropna(subset=['Rating'])

df['Genre'] = df['Genre'].fillna('Unknown')
df['Director'] = df['Director'].fillna('Unknown')
df['Actor 1'] = df['Actor 1'].fillna('Unknown')
df['Actor 2'] = df['Actor 2'].fillna('Unknown')
df['Actor 3'] = df['Actor 3'].fillna('Unknown')

df.reset_index(drop=True, inplace=True)

# Convert 'Votes' to numeric, handling commas
df['Votes'] = df['Votes'].str.replace(',', '').astype(float)
df['Votes'] = df['Votes'].fillna(0)

# Drop rows with missing 'Duration' or 'Year'
# df = df.dropna(subset=['Duration', 'Year'])

# Convert 'Duration' to numeric by removing 'min'
df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)

# Convert 'Year' to numeric
df['Year'] = df['Year'].str.replace(r'[()]', '', regex=True).astype(int)

# Combine actors into a single column
df['Actors'] = df['Actor 1'] + ', ' + df['Actor 2'] + ', ' + df['Actor 3']


# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[['Genre', 'Director', 'Actors']])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(['Genre', 'Director', 'Actors']))

# Combine encoded features with numerical features
X = pd.concat([df[['Year', 'Duration', 'Votes']], encoded_df], axis=1)
y = df['Rating']

print(X.shape)
print(y.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# standardize

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from xgboost import XGBRegressor

# Initialize and train the model
model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(y_pred)

# Save the model as a .pkl file
with open("MovieRating_saved.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model saved successfully!")








