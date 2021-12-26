from django.shortcuts import render
from django.http import HttpResponse

from matplotlib.gridspec import GridSpec
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
from time import sleep

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
import os
import twint

algorithm = 'mnb'
acc = '77'
file_name = 'static/models/'+algorithm.upper()+'_model_'+acc

# Create your views here.

def load_model():
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
        
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    
    tfidf = pickle.load(open(f"static/vector/tfidf_{algorithm}_{acc}","rb"))
    
    return model, tfidf, token

def extract_tweets(query, limit):
    # Configure
    c = twint.Config()
    c.Search = str(query)
    c.Lang = "en"
    c.Limit = limit
    c.Store_csv = True
    try:
        os.remove("static/data/tweets.csv")
        os.remove("static/data/analysis/tweets.csv")
        os.remove("static/data/analysis/predictions.csv")
        os.remove("static/data/analysis/merged.csv")
    except:
        pass
    c.Output = "static/data/tweets.csv"
    c.Pandas = True

    # Run
    twint.run.Search(c)
    
    return pd.read_csv(f'static/data/tweets.csv', delimiter=',')

def index(request):
    return render(request, './query.html')


def result(request):
    form = request.POST
    choice = form['choice']
    text = form['document']
    
    model, vector, token = load_model()
    
    if choice == 'Keywords':
        flag = True
        while flag:
            try:
                dataset = extract_tweets(text, 1000)
                flag = False
            except:
                sleep(1)
        # removing the unnecessary columns.
        dataset = dataset[['sentiment','tweet']]
        X = dataset['tweet'].fillna(' ')
        tweets = X

        y_pred = vector.transform(X)

        predictions = model.predict(y_pred)
        # saving tweets to csv
        tweets.to_csv(f'static/data/analysis/tweets.csv')
        # saving sentiment predictions to csv
        np.savetxt(f'static/data/analysis/predictions.csv',predictions, delimiter=',', fmt=('%s'))

        DATASET_COLUMNS  = ["sentiment"]

        # adding sentiment column to the beginning
        df = pd.read_csv(f'static/data/analysis/predictions.csv', header=None)
        df.rename(columns={0: 'sentiment'}, inplace=True)
        df.to_csv(f'static/data/analysis/predictions.csv', index=False) # save to new csv file

        # merging tweets and predictions
        filenames = [f'static/data/analysis/tweets.csv', f'static/data/analysis/predictions.csv']
        dfs = []
        for filename in filenames:
            # read the csv, making sure the first two columns are str
            df = pd.read_csv(filename, header=None, converters={0: str, 1: str})
            # change the column names so they won't collide during concatenation
            df.columns = [filename + str(cname) for cname in df.columns]
            dfs.append(df)

        # concatenate them horizontally
        merged = pd.concat(dfs,axis=1)
        # write it out
        merged.to_csv(f"static/data/analysis/merged.csv", header=None, index=None)

        df = pd.read_csv(f'static/data/analysis/merged.csv')

        labels = ['negative', 'positive']

        title_type = df.groupby('sentiment').agg('count')

        type_labels = ['positive', 'negative']
        type_counts = title_type.tweet.sort_values()

        colors = ['g', 'r']

        plt.subplot(aspect=1, title=f'Results')
        type_show_ids = plt.pie(type_counts, labels=type_labels, autopct='%1.1f%%', shadow=True, colors=colors)
        plt.savefig("static/img/sentiments.png", transparent=True)
        sentiment = 'static/img/sentiments.png'
        context = {
            'sentiment': sentiment,
        }
        return render(request, './tweets_result.html', context=context)
    analysis = vector.transform([text])
    
    polarity = model.predict(analysis)
    
    sentiment = "Positive" if polarity == 4 else "Negative"
    context = {
        'sentiment': sentiment,
    }
    return render(request, './text_result.html', context=context)