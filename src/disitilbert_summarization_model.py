#!pip install transformers
from transformers import pipeline
import pandas as pd
import os

# Load data
dir = os.getcwd()
train = pd.read_csv(dir+"/data/train.csv")

example = train.iloc[0]

# Define the text to be summarized
print("\nText: "+example.article)
# Define the label summary to be fulfiled
print("\nHighlight: "+example.highlights)

# Load the summarization pipeline
summarizer = pipeline(model='sshleifer/distilbart-cnn-12-6')


# Generate a summary
summary = summarizer(example.article, max_length=1024, min_length=10, do_sample=False)[0]['summary_text']

# Print the summary
print(summary)