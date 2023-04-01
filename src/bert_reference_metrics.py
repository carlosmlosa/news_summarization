from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score
from summarizer import Summarizer,TransformerSummarizer
import pandas as pd
import os

# https://pypi.org/project/bert-extractive-summarizer/


def calculate_scores(reference, candidate):
  # Calculate BLEU score
  bleu_score = sentence_bleu([reference.split()], candidate.split())
  

  # Calculate ROUGE scores
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
  scores = scorer.score(reference,candidate)
  

  # # Calculate Bert scores
  # P,R,F1= score([bert_summary], [highlights], lang="en", verbose=True)
  
  # bert_score = {"Precision":P, "Recall": R, "F1": F1}
  return bleu_score, scores["rouge1"].fmeasure, scores["rougeL"].fmeasure#, bert_score

def print_scores(bleu_score, rouge1,rougeL): #, bert_score
  print("BLEU score:", bleu_score)
  print("ROUGE-N score:",rouge1 )
  print("ROUGE-L score:",rougeL )
  # print(f"Bert score Precision:{{bert_score['Precision']}}, Recall: {{bert_score['Recall']}}, F1: {{bert_score['F1']}} ")


def evaluate(texts,model):
  results_df = pd.DataFrame(columns = ["Text","Highlights","Summary","BLEU","Rouge1","RougeL"])
  for i in range(len(texts)):
    summary = ''.join(model(texts.iloc[i].article, min_length=60,ratio=0.5))
    bleu,rougen,rougel=calculate_scores(summary,texts.iloc[i].highlights)
    results_df.loc[len(results_df)] = [texts.iloc[i].article,texts.iloc[i].highlights,summary,bleu,rougen,rougel]
    if i%20==1:
      print(results_df.iloc[-5:])
  return results_df


dir = os.getcwd()
test = pd.read_csv(dir+"/data/test.csv")




bert_model = Summarizer(hidden=[-1])
# bert_summary = ''.join(bert_model(text, min_length=60, max_length=240,ratio=0.5))
bert_reference_short = evaluate(test.iloc[:500],bert_model)
print(bert_reference_short.head())
bert_reference_short.to_csv(dir+"/data/reference_metrics.csv")

bert_reference_full = evaluate(test,bert_model)
print(bert_reference_full.head())
bert_reference_full.to_csv(dir+"/data/reference_metrics_full.csv")