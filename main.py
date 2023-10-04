import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Load data
train_data = pd.read_csv('../Ressources/train.csv')
test_data = pd.read_csv('../Ressources/test.csv')

