from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
# Load data
train_data = pd.read_csv('Ressources/train.csv')
test_data = pd.read_csv('Ressources/test.csv')
## Traitement des données
### On va utiliser les colonnes : 
### HomePlanet, CryoSleep, Destination, Age, VIP, Argent_Total

train_data = train_data.reindex (columns = ["PassengerId","Name","HomePlanet","Destination","CryoSleep","Cabin","Age","Classe_Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck", "Argent_Total", "Famille", "Transported"])
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
X = pd.get_dummies(train_data[features])
y = train_data["Transported"]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
forest_model = RandomForestClassifier(n_estimators= 2100, max_depth = 6, random_state=1)

# Entraînement des modèles
forest_model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
print("\nPredicting using RandomForest...")
forest_pred = forest_model.predict(X_test)
print("Done predicting using RandomForest.\n")
# Évaluation de RandomForest
print("Performance de RandomForest")
print("Précision :", accuracy_score(y_test, forest_pred))
print("Score F1 :", f1_score(y_test, forest_pred))
print("Matrice de confusion :\n", confusion_matrix(y_test, forest_pred))
