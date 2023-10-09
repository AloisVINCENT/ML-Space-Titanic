import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load data
train_data = pd.read_csv('Ressources/train.csv')
test_data = pd.read_csv('Ressources/test.csv')

# Mise en forme Age
train_data['Age2'] = 'Inconnu'
for i in range(len(train_data['Age'])):
    if (train_data['Age'][i] > 0 and train_data['Age'][i] < 18):
        train_data.loc[i, 'Age2'] = "Enfant"
    elif (train_data['Age'][i] >= 18 and train_data['Age'][i] < 30):
        train_data.loc[i, 'Age2'] = "Jeune"
    elif (train_data['Age'][i] >= 30 and train_data['Age'][i] < 50):
        train_data.loc[i, 'Age2'] = "Adulte"
    elif (train_data['Age'][i] >= 50):
        train_data.loc[i, 'Age2'] = "Vieux"
# Jeune      3375
# Adulte     2783
# Enfant     1367
# Vieux       811
# Inconnu     357

train_data["Argent_Total"] = train_data["RoomService"] + train_data["FoodCourt"] + train_data["ShoppingMall"] + train_data["Spa"] + train_data["VRDeck"]
# Ajout Colonne Famille
train_data['Famille'] = False
for i in range(len(train_data['Cabin'])-1):
    if (train_data['Cabin'][i] == train_data['Cabin'][i+1]):
        train_data.loc[i , "Famille"] = True


train_data["Argent_Total"] = train_data["RoomService"] + train_data["FoodCourt"] + train_data["ShoppingMall"] + train_data["Spa"] + train_data["VRDeck"]
for i in range(len(train_data['Age'])):
    if (train_data['Age2'][i] == "Inconnu"):
        rand = np.random.randint(1, 11)
        if (rand <= 4):
            train_data.loc[i, 'Age'] = np.random.randint(18, 30)
        elif (rand > 4 and rand <= 7):
            train_data.loc[i, 'Age'] = np.random.randint(30, 50)
        elif (rand > 7 and rand <= 9):
            train_data.loc[i, 'Age'] = np.random.randint(int(train_data['Age'].min()), 18)
        else:
            train_data.loc[i, 'Age'] = np.random.randint(50, int(train_data['Age'].max()))

for i in range(len(train_data['Age'])):
    if (train_data['Age'][i] > 0 and train_data['Age'][i] < 18):
        train_data.loc[i, 'Age2'] = "Enfant"
    elif (train_data['Age'][i] >= 18 and train_data['Age'][i] < 30):
        train_data.loc[i, 'Age2'] = "Jeune"
    elif (train_data['Age'][i] >= 30 and train_data['Age'][i] < 50):
        train_data.loc[i, 'Age2'] = "Adulte"
    elif (train_data['Age'][i] >= 50):
        train_data.loc[i, 'Age2'] = "Vieux"

train_data.to_csv('Ressources/train2.csv', index=False)
print("Done")