#Amazon data set VADER analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Iterable, Dict

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('vader_lexicon')

plt.style.use('ggplot')#

def read_data(filepath: str, num_rows: int) -> pd.DataFrame:
    data = pd.read_csv(filepath)
    data = data.head(num_rows)
    return data

def tokenize(text: str) -> Iterable[str]:
    tokens = nltk.word_tokenize(text)
    return tokens

def tagged(tokens: Iterable[str]) -> Iterable[tuple]:
    tagged = nltk.pos_tag(tokens)
    return tagged

def analyze_sentiment_for_given_txt(text: str) -> Dict[str, float]:
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    return sentiment_scores

def calculate_sentiment_scores(data: pd.DataFrame) -> pd.DataFrame:
    polarity = []
    sia = SentimentIntensityAnalyzer()

    for i in data['Id']:
        tmp = []
        tmp.append(i)
        polarity.append(tmp)

    q = 0
    for i in data['Text']:
        polarity[q].append(sia.polarity_scores(i))
        q += 1

    sentence_scores = {}
    for i in polarity:
        sentence_scores[i[0]] = i[1]

    sentence_scores = pd.DataFrame(sentence_scores).T
    sentence_scores = sentence_scores.reset_index().rename(columns={'index': 'Id'})
    sentence_scores = sentence_scores.merge(data, how='left')
    return sentence_scores

def scatter_plot(data: pd.DataFrame):
    plt.scatter(data['Score'], data['compound'], color='red', marker='.')
    plt.xlabel('Score')
    plt.ylabel('Compound')
    plt.title('VADER results by Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def bar_plot(data: pd.DataFrame):
    sentiment_means = data.groupby('Score')['compound'].mean()
    stars = sentiment_means.index
    plt.bar(stars, sentiment_means, align='center', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Average sentiment result')
    plt.title('Distributions of VADER results by Star Review')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def grouped_bar_plot(data: pd.DataFrame):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    sns.barplot(data=data, x=data['Score'], y=data['compound'], ax=axs[0, 0])
    sns.barplot(data=data, x=data['Score'], y=data['pos'], ax=axs[0, 1])
    sns.barplot(data=data, x=data['Score'], y=data['neu'], ax=axs[1, 0])
    sns.barplot(data=data, x=data['Score'], y=data['neg'], ax=axs[1, 1])
    axs[0, 0].set_title("Compound")
    axs[0, 1].set_title("Positive")
    axs[1, 0].set_title("Neutral")
    axs[1, 1].set_title("Negative")
    plt.show()

def main():
    data = read_data('./data/Reviews.csv', 1000)
    tokens = tokenize(data['Text'][50])
    print(tokens)
    print(tagged(tokens))
    sentence_scores = calculate_sentiment_scores(data)
    scatter_plot(sentence_scores)
    bar_plot(sentence_scores)
    grouped_bar_plot(sentence_scores)

if __name__ == "__main__":
    main()



#example for seperate sentence
'''
#NLTK
#creating tokens
example = data['Text'][50]
#print(example)
tokens = nltk.word_tokenize(example)
#print(tokens[:10])
#finding part of speach for each token
tagged = nltk.pos_tag(tokens)
#print(tagged)
#putting into entities
entities = nltk.chunk.ne_chunk(tagged_tokens = tagged)

#VADER
#each word is scored and combined to a total score

#creating sentiment analysier object

sia = SentimentIntensityAnalyzer()

polar = sia.polarity_scores("I am so happy")
print(polar)
polar = sia.polarity_scores("I hate all my friends")
print(polar)
polar = sia.polarity_scores(example)
print(polar)'''



