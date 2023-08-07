import joblib

#load trained model
model = joblib.load('house_value_model.pkl')

house_1 = [
    4000, #size in square feet
    3, # number of bedrooms
    2, #number of bathrooms
]


homes = [
    house_1,
]

home_values = model.predict(homes)

#grab first prediction
predicted_value = home_values[0]
print("House details:")
print(f"- {house_1[0]} sq feet")
print(f"- {house_1[1]} bedrooms")
print(f"- {house_1[2]} bedrooms")
print(f"Estimated value: ${predicted_value:,.2f}")

