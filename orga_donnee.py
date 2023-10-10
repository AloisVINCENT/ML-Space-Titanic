import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def orga_data (train_data, test_data):
# *----------------------------------------------------------------------------------*
# *----------------------------Partie Train data-------------------------------------*
# *----------------------------------------------------------------------------------*

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
    # train_data['Famille'] = False
    # for i in range(len(train_data['Cabin'])-1):
    #     if (train_data['Cabin'][i] == train_data['Cabin'][i+1]):
    #         train_data.loc[i , "Famille"] = True


    train_data["Argent_Total"] = train_data["RoomService"] + train_data["FoodCourt"] + train_data["ShoppingMall"] + train_data["Spa"] + train_data["VRDeck"]
    for i in range(len(train_data['Age'])):
        if (train_data['Classe_Age'][i] == "Inconnu"):
            rand = np.random.randint(1, 11)
            if (rand <= 4):
                train_data.loc[i, 'Age'] = np.random.randint(18, 30)
                train_data.loc[i, 'Classe_Age'] = "Jeune"
            elif (rand > 4 and rand <= 7):
                train_data.loc[i, 'Age'] = np.random.randint(30, 50)
                train_data.loc[i, 'Classe_Age'] = "Adulte"
            elif (rand > 7 and rand <= 9):
                train_data.loc[i, 'Age'] = np.random.randint(int(train_data['Age'].min()), 18)
                train_data.loc[i, 'Classe_Age'] = "Enfant"
            else:
                train_data.loc[i, 'Age'] = np.random.randint(50, int(train_data['Age'].max()))
                train_data.loc[i, 'Classe_Age'] = "Vieux"

    train_data['CryoSleep'].fillna(False)
    train_data['Destination'].fillna("TRAPPIST-1e", inplace = True)
    train_data['VIP'].fillna(train_data['VIP'].median(), inplace = True)
    train_data['HomePlanet'].fillna("Earth", inplace = True)
    train_data["Argent_Total"].fillna(0, inplace= True)

    train_data = train_data.reindex (columns = ["PassengerId","Name","HomePlanet","Destination","CryoSleep","Cabin","Age","Classe_Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck", "Argent_Total", "Transported"])
    print("Done pour train data")

    # *----------------------------------------------------------------------------------*
    # *----------------------------Partie Test data--------------------------------------*
    # *----------------------------------------------------------------------------------*

    test_data['Classe_Age'] = 'Inconnu'
    for i in range(len(test_data['Age'])):
        if (test_data['Age'][i] > 0 and test_data['Age'][i] < 18):
            test_data.loc[i, 'Classe_Age'] = "Enfant"
        elif (test_data['Age'][i] >= 18 and test_data['Age'][i] < 30):
            test_data.loc[i, 'Classe_Age'] = "Jeune"
        elif (test_data['Age'][i] >= 30 and test_data['Age'][i] < 50):
            test_data.loc[i, 'Classe_Age'] = "Adulte"
        elif (test_data['Age'][i] >= 50):
            test_data.loc[i, 'Classe_Age'] = "Vieux"

    # Classe_Age
    # Jeune      1766
    # Adulte     1325
    # Enfant      637
    # Vieux       376
    # Inconnu     173

    test_data["RoomService"].fillna(test_data["RoomService"].median(), inplace= True)
    test_data["FoodCourt"].fillna(test_data["FoodCourt"].median(), inplace= True)
    test_data["ShoppingMall"].fillna(test_data["ShoppingMall"].median(), inplace= True)
    test_data["Spa"].fillna(test_data["Spa"].median(), inplace= True)
    test_data["VRDeck"].fillna(test_data["VRDeck"].median(), inplace= True)

    test_data["Argent_Total"] = test_data["RoomService"] + test_data["FoodCourt"] + test_data["ShoppingMall"] + test_data["Spa"] + test_data["VRDeck"]
    # Ajout Colonne Famille
    # test_data['Famille'] = False
    # for i in range(len(test_data['Cabin'])-1):
    #     if (test_data['Cabin'][i] == test_data['Cabin'][i+1]):
    #         test_data.loc[i , "Famille"] = True

    for i in range(len(test_data['Age'])):
        if (test_data['Classe_Age'][i] == "Inconnu"):
            rand = np.random.randint(1, 11)
            if (rand <= 4):
                test_data.loc[i, 'Age'] = np.random.randint(18, 30)
                test_data.loc[i, 'Classe_Age'] = "Jeune"
            elif (rand > 4 and rand <= 7):
                test_data.loc[i, 'Age'] = np.random.randint(30, 50)
                test_data.loc[i, 'Classe_Age'] = "Adulte"
            elif (rand > 7 and rand <= 9):
                test_data.loc[i, 'Age'] = np.random.randint(int(test_data['Age'].min()), 18)
                test_data.loc[i, 'Classe_Age'] = "Enfant"
            else:
                test_data.loc[i, 'Age'] = np.random.randint(50, int(test_data['Age'].max()))
                test_data.loc[i, 'Classe_Age'] = "Vieux"

    test_data['CryoSleep'].fillna(False)
    test_data['Destination'].fillna("TRAPPIST-1e", inplace = True)
    test_data['VIP'].fillna(test_data['VIP'].median(), inplace = True)
    test_data['HomePlanet'].fillna("Earth", inplace = True)
    test_data["Argent_Total"].fillna(0, inplace= True)

    test_data = test_data.reindex (columns = ["PassengerId","Name","HomePlanet","Destination","CryoSleep","Cabin","Age","Classe_Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck", "Argent_Total", "Transported"])
    print("Done pour test data")
    return train_data, test_data