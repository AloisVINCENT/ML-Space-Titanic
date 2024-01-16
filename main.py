import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load data
train_data = pd.read_csv('Ressources/train.csv')
test_data = pd.read_csv('Ressources/test.csv')

## Traitement des données
### On va utiliser les colonnes : 
### HomePlanet, CryoSleep, Destination, Age, VIP, Argent_Total
features = ["Age", "CryoSleep", "Destination", "VIP", "HomePlanet"]
for feature in features :
    train_data[feature] = train_data[feature].bfill()
    test_data[feature] = test_data[feature].bfill()

## Création du modèle de Forêt
forest_model = RandomForestClassifier(n_estimators= 2100, max_depth = 6, random_state=1)
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

## Creation du modele DecisionTree
decision_model = DecisionTreeClassifier(max_depth= 6)

print("\nProcessing DecisionTreeClassifier...")
decision_model.fit(X,y)
decision_pred = decision_model.predict(X_test)
print("Done DecisionTree\n")

output2 = pd.DataFrame({"PassengerId" : test_data.PassengerId, "Transported" : decision_pred})
print("\n Nombre de valeur différentes Forest")
print(output["Transported"].value_counts())
print("\n Nombre de valeur différentes Decision")
print(output2["Transported"].value_counts())
output2.to_csv("submission2.csv", index = False)
print("Your second submission was successfully saved!")