from keras.models import load_model
import joblib

#load trained model
model = load_model('house_value_model.h5')

#load the data scalers so that we can transform
#new data and prediction the same way as 
#the training data


X_scaler = joblib.load('X_scalar.pkl')
y_scaler = joblib.load('y_scalar.pkl')

#define house we want to predict
house_1 = [
    2000, #size in sq feet
    3, 
    2,
]

#put single house in array
homes = [
    house_1
]

#scale the data like the training data
scaled_home_data = X_scaler.transform(homes)

#make a prediction for each house in the homes
#array (but we only have one)
home_values = model.predict(scaled_home_data)

#the prediction from nn will scale from 0 to 1 
# just like training data

unscaled_home_values = y_scaler.inverse_transform(
    home_values
)

#predicting only the price of one house
#grab first prediction 

predicted_value = unscaled_home_values[0][0]

#print results

print("house details:")
print(f"-{house_1[0]} sq feet")
print(f"-{house_1[1]} sq feet")
print(f"-{house_1[2]} sq feet")
print(f"Estimated value: ${predicted_value:,.2f}")



