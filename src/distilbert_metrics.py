#!pip install transformers
from transformers import pipeline
import pandas as pd
import os
from bert_reference_metrics import calculate_scores
import re


def evaluate_distilBERT(texts,model):
    """Function that summarizes texts and computes the metrics"""
    results_df = pd.DataFrame(columns = ["Text","Highlights","Summary","BLEU","Rouge1","RougeL"])
    for i in range(len(texts)):
        # Split the text into segments of max length 1024
        sentences = re.split('(?<=\.) ', str(texts.iloc[i].article))
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
            summary = model(segment, max_length=1024, min_length=10, do_sample=False)[0]['summary_text']
            summaries.append(summary)

        # Combine the summaries into a final summary
        final_summary = " ".join(summaries)
        bleu,rougen,rougel=calculate_scores(final_summary,texts.iloc[i].highlights)
        results_df.loc[len(results_df)] = [texts.iloc[i].article,texts.iloc[i].highlights,final_summary,bleu,rougen,rougel]
        if i%20==1:
            print(results_df.iloc[-5:])
    return results_df


if __name__ == "__main__":
    # Load data
    dir = os.getcwd()
    test = pd.read_csv(dir+"/data/test.csv")

    # Load the summarization pipeline
    distilbert_summarizer_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", tokenizer="sshleifer/distilbart-cnn-12-6")

    # Evaluate distilbert model with test dataset
    distilbert_test = evaluate_distilBERT(test,distilbert_summarizer_model)

    # Save results
    print(distilbert_test.head())
    distilbert_test.to_csv(dir+"/data/distilbert_test_metrics.csv")


