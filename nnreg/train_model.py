import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.models import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
from keras.layers import Activation,Dense

#load data
df =pd.read_csv('house_data.csv')

X  = df[["sq_feet","num_bedrooms", "num_bathrooms"]]
y = df[["sale_price"]]

#scale inputs between 0 and 1
X_scalar = MinMaxScaler(feature_range=(0,1))
y_scalar = MinMaxScaler(feature_range=(0,1))

#scale the input and output data
X[X.columns] = X_scalar.fit_transform(X[X.columns])
y[y.columns] = y_scalar.fit_transform(y[y.columns])

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25
)

model = Sequential()

model.add(Dense(50, input_dim =3, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(
    loss = 'mean_squared_error',
    optimizer = 'SGD'
)

model.fit(
    X_train, 
    y_train, 
    epochs =50,
    batch_size=8,
    shuffle=True,
    verbose=2
)

#save scalars to file
joblib.dump(X_scalar, "X_scalar.pkl")
joblib.dump(y_scalar, "y_scalar.pkl")

model.save("house_value_model.h5")

print("Model training results:")

#report mean abs. erro on training
#set in value scaled back to $$
predictions_train = model.predict(X_train, verbose=0)

mse_train= mean_absolute_error(
    y_scalar.inverse_transform(predictions_train),
    y_scalar.inverse_transform(y_train)
)
print(f" - Training Set Error: {mse_train}")

#Report mean abs. error on test set in $
predictions_test= model.predict(X_test, verbose=0)

mse_test = mean_absolute_error(
    y_scalar.inverse_transform(predictions_test),
    y_scalar.inverse_transform(y_test)
)
print(f" - Test Set Error: {mse_test}")