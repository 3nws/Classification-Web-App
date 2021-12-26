from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from wsgiref.util import FileWrapper

from matplotlib.gridspec import GridSpec
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
from time import sleep
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
import os
import twint
import mimetypes

algorithm = 'mnb'
acc = '77'
file_name = 'static/models/'+algorithm.upper()+'_model_'+acc

# Create your views here.

algorithms = {
    'mnb': MultinomialNB(),
    'lrg': LogisticRegression(random_state=0, class_weight='balanced'),
    'svc': LinearSVC(),
}

months = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12,
}

def download_model(request):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = 'model'
    filepath = base_dir + '/static/trained_models/' + filename
    thefile = filepath
    filename = os.path.basename(thefile)
    response = StreamingHttpResponse(FileWrapper(open(thefile, 'rb')), content_type=mimetypes.guess_type(thefile)[0])
    response['Content-Length'] = os.path.getsize(thefile)
    response['Content-Disposition'] = "Attachment;filename=%s" % filename
    return response

def download_vector(request):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = 'tfidf_vectorizer'
    filepath = base_dir + '/static/fitted_vector/' + filename
    thefile = filepath
    filename = os.path.basename(thefile)
    response = StreamingHttpResponse(FileWrapper(open(thefile, 'rb')), content_type=mimetypes.guess_type(thefile)[0])
    response['Content-Length'] = os.path.getsize(thefile)
    response['Content-Disposition'] = "Attachment;filename=%s" % filename
    return response
    
def load_model():
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
        
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    
    tfidf = pickle.load(open(f"static/vector/tfidf_{algorithm}_{acc}","rb"))
    
    return model, tfidf, token

def extract_tweets(query, limit, year, month):
    # Configure
    c = twint.Config()
    c.Search = str(query)
    c.Lang = "en"
    if year:
        month = months[month]
        c.Since = f"{year}-{month}-1"
        c.Until = f"{year}-{str(month+1)}-1" if month != 12 else f"{str(year+1)}-1-1"
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

def train(request):
    return render(request, './train.html')

def train_result(request):
    DATASET_ENCODING = "ISO-8859-1"
    dataset = pd.read_csv(request.FILES['csv_file'], delimiter=',', encoding=DATASET_ENCODING)
    form = request.POST
    text = form['column_1']
    target = form['column_2']
    max_features = int(form['max_features'])
    test_size = float(form['test-size'])
    algorithm = str(form['algo'])
    
    dataset = dataset[[text,target]]
    dataset.drop_duplicates()
    
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')

    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=(1,2), tokenizer=token.tokenize)

    X = dataset[text]

    X = tfidf.fit_transform(dataset[text])

    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=5)
    
    # creating the model using multinomial naive bayes algorithm
    model = algorithms[algorithm]

    # training the model
    model.fit(X_train, y_train)

    # testing our predictions
    y_pred = model.predict(X_test)

    acc = classification_report(y_test, y_pred, output_dict=True)
    
    # exporting the model and the trained vectorizer
    pickle.dump(model, open('static/trained_models/model', 'wb'))
    pickle.dump(tfidf, open('static/fitted_vector/tfidf_vectorizer', 'wb'))
    
    with open('static/trained_models/model', 'rb') as f:
        model = pickle.load(f)
    
    tfidf = pickle.load(open(f'static/fitted_vector/tfidf_vectorizer',"rb"))
    
    context = {
        'acc': int((float(acc.get("accuracy"))*100)),
        'model': model,
        'vector': tfidf,
    }
    
    return render(request, './train_result.html', context=context)

def result(request):
    form = request.POST
    choice = form['choice']
    text = form['document']
    year = form['year']
    month_name = form['month']
    
    if year == 'Year' or month_name == 'Month':
        year = None
    
    model, vector, token = load_model()
    
    if choice == 'Keywords':
        flag = True
        while flag:
            try:
                dataset = extract_tweets(text, 100, year, month_name)
                flag = False
            except Exception as e:
                print(e)
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
        if year:
            plt.subplot(aspect=1, title=f'Results for {month_name} in {year}')
        else:
            plt.subplot(aspect=1, title=f'Results')
        type_show_ids = plt.pie(type_counts, labels=type_labels, autopct='%1.1f%%', shadow=True, colors=colors)
        plt.savefig("static/img/sentiments.png", transparent=True)
        return render(request, './tweets_result.html')
    analysis = vector.transform([text])
    
    polarity = model.predict(analysis)
    
    sentiment = "Positive" if polarity == 4 else "Negative"
    context = {
        'sentiment': sentiment,
    }
    return render(request, './text_result.html', context=context)