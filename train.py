import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("house.csv")

X = df[[
    'Avg. Area Income',
    'Avg. Area House Age',
    'Avg. Area Number of Rooms',
    'Avg. Area Number of Bedrooms',
    'Area Population'
]]

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

comparison = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': predictions
})

print(comparison.head())

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

new_house = [[65000, 5, 7, 4, 30000]]

predicted_price = model.predict(new_house)

print("Predicted Price:", predicted_price[0])

import pickle

pickle.dump(model, open("house_model.pkl", "wb"))

print("Model Saved Successfully")

income = float(input("Enter Avg. Area Income: "))
age = float(input("Enter Avg. Area House Age: "))
rooms = float(input("Enter Avg. Area Number of Rooms: "))
bedrooms = float(input("Enter Avg. Area Number of Bedrooms: "))
population = float(input("Enter Area Population: "))

new_house = [[income, age, rooms, bedrooms, population]]

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])