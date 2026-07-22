import pickle

model = pickle.load(open("house_model.pkl", "rb"))

new_house = [[65000, 5, 7, 4, 30000]]

price = model.predict(new_house)

print(price)