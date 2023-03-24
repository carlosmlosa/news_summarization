import pandas as pd
import os
dir = os.getcwd()

train = pd.read_csv(dir+"/data/train.csv")
test = pd.read_csv(dir+"/data/test.csv")
validation = pd.read_csv(dir+"/data/validation.csv")
print(train.head())
print(test.head())
print(validation.head())