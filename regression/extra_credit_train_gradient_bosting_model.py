import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib

df = pd.read_csv("house_data.csv")

#features used for predictions
X = df[["sq_feet", "num_bedrooms", "num_bathrooms"]]

#matching expect output
y = df["sale_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25
)

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

joblib.dump(model, "house_value_model.pkl")

print("Model training results:")

#report an error rate on the training set
mse_train = mean_absolute_error (
    y_train, 
    model.predict(X_train)
)

print(f" -- Training Set Error: {mse_train}")


#report an error rate on the test set
mse_test = mean_absolute_error (
    y_test, 
    model.predict(X_test)
)

print(f" -- Training Set Error: {mse_test}")