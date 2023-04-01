import pandas as pd
import os
dir = os.getcwd()

# train = pd.read_csv(dir+"/data/train.csv")
# test = pd.read_csv(dir+"/data/test.csv")
# validation = pd.read_csv(dir+"/data/validation.csv")
# print(train.head())
# print(test.head())
# print(validation.head())

reference_metrics = pd.read_csv(dir+"/data/reference_metrics_full.csv")
print(reference_metrics.describe())
# print("\nText\n",reference_metrics.iloc[2050].Text)
# print("\nSummary\n",reference_metrics.iloc[2050].Summary)