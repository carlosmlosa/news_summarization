#!pip install transformers
from transformers import pipeline
import pandas as pd
import os
import re

# Load data
dir = os.getcwd()
train = pd.read_csv(dir+"/data/train.csv")

example = train.iloc[0]

# Define the text to be summarized
print("\nText: "+example.article)
# Define the label summary to be fulfiled
print("\nHighlight: "+example.highlights)

# Load the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", tokenizer="sshleifer/distilbart-cnn-12-6")


# Split the text into segments of max length 1024
sentences = re.split('(?<=\.) ', example.article)
segments = []
current_segment = ""
for sentence in sentences:
    if len(current_segment + sentence) > 4096:
        segments.append(current_segment.strip())
        current_segment = sentence
    else:
        current_segment += sentence
segments.append(current_segment.strip())


# Generate a summary for each segment
summaries = []
for segment in segments:
    summary = summarizer(segment, max_length=1024, min_length=10, do_sample=False)[0]['summary_text']
    summaries.append(summary)

# Combine the summaries into a final summary
final_summary = " ".join(summaries)

# Print the final summary
print(final_summary)
print(summarizer(final_summary, max_length=512, min_length=10, do_sample=False)[0]['summary_text'])