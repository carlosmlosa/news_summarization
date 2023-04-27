import pandas as pd
import os
dir = os.getcwd()
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

def box_plot_metric(bert,distilbart,metric):
    fig = go.Figure()
    fig.add_trace(go.Box(y=bert[metric],name = "BERT"))
    fig.add_trace(go.Box(y=distilbart[metric],name = "distilbart"))
    fig.update_layout(title_text=metric)
    return fig

def plot_hist(bert,distilbart,metric):

    fig = go.Figure()
    # fig.add_trace(go.Histogram(x=reference.BLEU))
    fig.add_trace(go.Histogram(x=bert[metric],name="BERT"))
    fig.add_trace(go.Histogram(x=distilbart[metric],name="Distilbart"))

    # Overlay both histograms
    fig.update_layout(barmode='overlay',
                    title_text=metric, # title of plot
                    xaxis_title_text='Value', # xaxis label
                    yaxis_title_text='Count')# yaxis label)) 
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.6)
    return fig


distilbart = pd.read_csv(dir+"/data/distilbart_test_metrics.csv",index_col=0)
bert = pd.read_csv(dir+"/data/reference_metrics_full.csv",index_col=0)
# print(distilbart.describe())
# print(reference.describe())

# fig = px.histogram(reference, x="Rouge1")
# fig.show()








# df = reference.merge(distilbart,left_on="Text",right_on="Text",suffixes=("_BERT","_distilbart"))
# print(df)
# fig = px.box(df, x="Text", y="RougeL_BERT")


figures =[]

hist_BLEU = plot_hist(bert,distilbart,"BLEU")
figures.append((hist_BLEU,"hist_BLEU"))
hist_rouge1 = plot_hist(bert,distilbart,"Rouge1")
figures.append((hist_rouge1,"hist_rouge1"))
hist_rougeL = plot_hist(bert,distilbart,"RougeL")
figures.append((hist_rougeL,"hist_rougeL"))
box_BLEU = box_plot_metric(bert,distilbart,metric="BLEU")
figures.append((box_BLEU,"box_BLEU"))
box_rouge1 = box_plot_metric(bert,distilbart,metric="Rouge1")
figures.append((box_rouge1,"box_rouge1"))
box_rougeL = box_plot_metric(bert,distilbart,metric="RougeL")
figures.append((box_rougeL,"box_rougeL"))

for fig in figures:
    fig[0].write_image(dir+"/fig/"+fig[1]+".jpeg")
    # fig[0].show()

print(bert.describe())
print(distilbart.describe())


