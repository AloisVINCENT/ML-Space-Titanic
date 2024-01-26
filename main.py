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
col_to_sum = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
for col in col_to_sum:
    train_data[col] = train_data[col].fillna(0)
    test_data[col] = test_data[col].fillna(0)
    train_data[col] = train_data[col].astype(int)
    test_data[col] = test_data[col].astype(int)
train_data["Argent_Total"] = train_data[col_to_sum].sum(axis=1)
features = ["Age", "CryoSleep", "Destination", "VIP", "HomePlanet", "Argent_Total"]
train_data = train_data.reindex (columns = ["PassengerId","Name","HomePlanet","Destination","CryoSleep","Cabin","Age","Classe_Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck", "Argent_Total", "Famille", "Transported"])
test_data = test_data.reindex (columns = ["PassengerId","Name","HomePlanet","Destination","CryoSleep","Cabin","Age","Classe_Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck", "Argent_Total", "Famille"])

train_data[features] = train_data[features].fillna(0)
test_data[features] = test_data[features].fillna(0)


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