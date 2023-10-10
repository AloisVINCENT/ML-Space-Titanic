import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from orga_donnee import orga_data

# Load data
train_data = pd.read_csv('Ressources/train.csv')
test_data = pd.read_csv('Ressources/test.csv')

## Traitement des données
### On va utiliser les colonnes : 
### HomePlanet, CryoSleep, Destination, Age, VIP, Argent_Total

train_data, test_data = orga_data(train_data, test_data)
train_data.to_csv('Ressources/train2.csv', index=False)
test_data.to_csv('Ressources/test2.csv', index=False)

## Création du modèle de Forêt
forest_model = RandomForestClassifier(n_estimators= 2100, max_depth = 6, random_state=1)

features = ["Age", "CryoSleep", "Destination", "VIP", "HomePlanet"] 

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
