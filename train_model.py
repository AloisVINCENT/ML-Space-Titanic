from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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

for i in train_data["Age"]:
    if (i < 18):
        train_data.loc[i, "Classe_Age"] = "Enfant"
    elif (i >= 18 or i <= 63):
        train_data.loc[i, "Classe_Age"] = "Adulte"
    else:
        train_data.loc[i, "Classe_Age"] = "Vieux"


# X = pd.get_dummies(train_data[features])
# y = train_data["Transported"]


# # Division des données en ensembles d'entraînement et de test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# forest_model = RandomForestClassifier(n_estimators= 2100, max_depth = 6, random_state=1)
# decision_model = DecisionTreeClassifier(max_depth= 6)

# # Entraînement des modèles
# forest_model.fit(X_train, y_train)
# decision_model.fit(X_train, y_train)

# # Prédiction sur l'ensemble de test
# forest_pred = forest_model.predict(X_test)
# decision_pred = decision_model.predict(X_test)

# # Évaluation de RandomForest
# print("Performance de RandomForest")
# print("Précision :", accuracy_score(y_test, forest_pred))
# print("Score F1 :", f1_score(y_test, forest_pred))
# print("Matrice de confusion :\n", confusion_matrix(y_test, forest_pred))

# # Évaluation de DecisionTree
# print("\nPerformance de DecisionTree")
# print("Précision :", accuracy_score(y_test, decision_pred))
# print("Score F1 :", f1_score(y_test, decision_pred))
# print("Matrice de confusion :\n", confusion_matrix(y_test, decision_pred))
