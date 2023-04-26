import pandas as pd
import os
dir = os.getcwd()

# train = pd.read_csv(dir+"/data/train.csv")
# test = pd.read_csv(dir+"/data/test.csv")
# validation = pd.read_csv(dir+"/data/validation.csv")
# print(train.head())
# print(test.head())
# print(validation.head())

reference_metrics = pd.read_csv(dir+"/data/reference_metrics_full.csv",index_col=0)
sample = reference_metrics.iloc[1]
print("\nText: \n",sample.Text)
print("\nHighlights: \n",sample.Highlights)
print("\nSummary: \n", sample.Summary)
print(sample)

