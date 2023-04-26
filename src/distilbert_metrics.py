#!pip install transformers
from transformers import pipeline
import pandas as pd
import os
from bert_reference_metrics import print_scores,calculate_scores


def evaluate_distilBERT(texts,model):
    results_df = pd.DataFrame(columns = ["Text","Highlights","Summary","BLEU","Rouge1","RougeL"])
    for i in range(len(texts)):
        summary = model(texts.iloc[i].article, max_length=1024,min_length=10,do_sample=False)[0]['summary_text']
        bleu,rougen,rougel=calculate_scores(summary,texts.iloc[i].highlights)
        results_df.loc[len(results_df)] = [texts.iloc[i].article,texts.iloc[i].highlights,summary,bleu,rougen,rougel]
        if i%20==1:
            print(results_df.iloc[-5:])
    return results_df


if __name__ == "__main__":
    # Load data
    dir = os.getcwd()
    test = pd.read_csv(dir+"/data/test.csv")

    # Load the summarization pipeline
    distilbert_summarizer_model = pipeline(model='sshleifer/distilbart-cnn-12-6')

    distilbertbert_test1 = evaluate_distilBERT(test,distilbert_summarizer_model)
    print(distilbertbert_test1.head())
    distilbertbert_test1.to_csv(dir+"/data/distilbert_test.csv")


