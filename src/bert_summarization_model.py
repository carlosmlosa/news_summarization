# Reference code https://medium.com/analytics-vidhya/text-summarization-using-bert-gpt2-xlnet-5ee80608e961

from summarizer import Summarizer
import pandas as pd
import os



dir = os.getcwd()

train = pd.read_csv(dir+"/data/train.csv")


example = train.iloc[0]
body = example.article
print("\nText: "+body)
print("\nHighlight: "+example.highlights)
bert_model = Summarizer() 
bert_summary = ''.join(bert_model(body, min_length=20,max_length=1000))
print("\nModel summary: "+bert_summary)