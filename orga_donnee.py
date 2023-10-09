import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load data
train_data = pd.read_csv('Ressources/train.csv')
test_data = pd.read_csv('Ressources/test.csv')

# Mise en forme Age
train_data['Classe_Age'] = 'Inconnu'
for i in range(len(train_data['Age'])):
    if (train_data['Age'][i] > 0 and train_data['Age'][i] < 18):
        train_data.loc[i, 'Classe_Age'] = "Enfant"
    elif (train_data['Age'][i] >= 18 and train_data['Age'][i] < 30):
        train_data.loc[i, 'Classe_Age'] = "Jeune"
    elif (train_data['Age'][i] >= 30 and train_data['Age'][i] < 50):
        train_data.loc[i, 'Classe_Age'] = "Adulte"
    elif (train_data['Age'][i] >= 50):
        train_data.loc[i, 'Classe_Age'] = "Vieux"
# Jeune      3375
# Adulte     2783
# Enfant     1367
# Vieux       811
# Inconnu     357

train_data["RoomService"].fillna(train_data["RoomService"].median(), inplace= True)
train_data["FoodCourt"].fillna(train_data["FoodCourt"].median(), inplace= True)
train_data["ShoppingMall"].fillna(train_data["ShoppingMall"].median(), inplace= True)
train_data["Spa"].fillna(train_data["Spa"].median(), inplace= True)
train_data["VRDeck"].fillna(train_data["VRDeck"].median(), inplace= True)

train_data["Argent_Total"] = train_data["RoomService"] + train_data["FoodCourt"] + train_data["ShoppingMall"] + train_data["Spa"] + train_data["VRDeck"]
# Ajout Colonne Famille
train_data['Famille'] = False
for i in range(len(train_data['Cabin'])-1):
    if (train_data['Cabin'][i] == train_data['Cabin'][i+1]):
        train_data.loc[i , "Famille"] = True


train_data["Argent_Total"] = train_data["RoomService"] + train_data["FoodCourt"] + train_data["ShoppingMall"] + train_data["Spa"] + train_data["VRDeck"]
for i in range(len(train_data['Age'])):
    if (train_data['Classe_Age'][i] == "Inconnu"):
        rand = np.random.randint(1, 11)
        if (rand <= 4):
            train_data.loc[i, 'Age'] = np.random.randint(18, 30)
        elif (rand > 4 and rand <= 7):
            train_data.loc[i, 'Age'] = np.random.randint(30, 50)
        elif (rand > 7 and rand <= 9):
            train_data.loc[i, 'Age'] = np.random.randint(int(train_data['Age'].min()), 18)
        else:
            train_data.loc[i, 'Age'] = np.random.randint(50, int(train_data['Age'].max()))

train_data = train_data.reindex (columns = ["PassengerId","Name","HomePlanet","Destination","CryoSleep","Cabin","Age","Classe_Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck", "Argent_Total", "Famille", "Transported"])
train_data.to_csv('Ressources/train2.csv', index=False)
print("Done")