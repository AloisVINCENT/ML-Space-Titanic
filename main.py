import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Load data
train_data = pd.read_csv('Ressources/train.csv')
test_data = pd.read_csv('Ressources/test.csv')

## Traitement des données
### On va utiliser les colonnes : 
### HomePlanet, CryoSleep, Destination, Age, VIP, Argent_Total


train_data["Argent_Total"] = train_data["RoomService"] + train_data["FoodCourt"] + train_data["ShoppingMall"] + train_data["Spa"] + train_data["VRDeck"]
train_data['Age'].fillna(train_data['Age'].median(), inplace= True)
train_data['CryoSleep'].fillna(False)
train_data['Destination'].fillna("TRAPPIST-1e", inplace = True)
train_data['VIP'].fillna(train_data['VIP'].median(), inplace = True)
train_data['HomePlanet'].fillna("Earth", inplace = True)
train_data["Argent_Total"].fillna(0, inplace= True)

test_data["Argent_Total"] = test_data["RoomService"] + test_data["FoodCourt"] + test_data["ShoppingMall"] + test_data["Spa"] + test_data["VRDeck"]
test_data["Argent_Total"].fillna(0, inplace= True)
test_data['Age'].fillna(test_data['Age'].median(), inplace= True)
test_data['CryoSleep'].fillna(False)
test_data['Destination'].fillna("TRAPPIST-1e", inplace = True)
test_data['VIP'].fillna(test_data['VIP'].median(), inplace = True)
test_data['HomePlanet'].fillna("Earth", inplace = True)

print(train_data.head())

## Création du modèle de Forêt
forest_model = RandomForestClassifier(n_estimators= 5000, max_depth = 6, random_state=1)

features = ["Argent_Total", "Age", "CryoSleep", "Destination", "VIP", "HomePlanet"] 

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

y = train_data["Transported"]

print("\nPredicting using RandomForest...")
forest_model.fit(X, y)
forest_pred = forest_model.predict(X_test)
print("Done predicting using RandomForest.\n")


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': forest_pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
