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

plt.style.use('ggplot')
sia = SentimentIntensityAnalyzer()

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
    #calculating sentiment scores for 'Text' column
    data['sentiment_scores'] = data['Text'].apply(sia.polarity_scores)

    sentiment_cols = data['sentiment_scores'].apply(pd.Series)
    data = pd.concat([data, sentiment_cols], axis=1)

    data.drop(columns=['sentiment_scores'], inplace=True)

    return data


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

def map_to_sentiment_category3(score: int):
    if score in [1, 2]:
        return -1  #Negative
    elif score == 3:
        return 0   #Neutral
    elif score in [4, 5]:
        return 1   #Positive

def map_compound_to_sentiment_category3(compound_score: float):
    if compound_score <= -0.25:
        return -1  #Negative
    elif compound_score >= 0.25:
        return 1   #Positive
    else:
        return 0   #Neutral

def map_to_sentiment_category2(score: int):
    if score in [1, 2, 3]:
        return -1  #Negative
    elif score in [4, 5]:
        return 1   #Positive

def map_compound_to_sentiment_category2(compound_score: float):
    if compound_score < 0:
        return -1  #Negative
    else:
        return 1   #Positive

#Creating csv file to compare predictions with actual data
def create_csv(data: pd.DataFrame, size: int):
    if size == 3:
        data['actual_sentiment'] = data['Score'].apply(map_to_sentiment_category3)
        data['predicted_sentiment'] = data['compound'].apply(map_compound_to_sentiment_category3)
    elif size == 2:
        data['actual_sentiment'] = data['Score'].apply(map_to_sentiment_category2)
        data['predicted_sentiment'] = data['compound'].apply(map_compound_to_sentiment_category2)
    #Create final csv and save
    result_data = data[['actual_sentiment', 'predicted_sentiment']]
    result_data.to_csv(f'./sentiment_results{size}.csv', index=False)

    return result_data

def calculate_precision(actual_sentiment, predicted_sentiment, label: int):
    TP = sum((actual == label) and (predicted == label) for actual, predicted in zip(actual_sentiment, predicted_sentiment))
    FP = sum((actual != label) and (predicted == label) for actual, predicted in zip(actual_sentiment, predicted_sentiment))

    precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    return precision

def calculate_recall(actual_sentiment: pd.DataFrame, predicted_sentiment: pd.DataFrame, label: int):
    TP = sum((actual == label) and (predicted == label) for actual, predicted in zip(actual_sentiment, predicted_sentiment))
    FN = sum((actual == label) and (predicted != label) for actual, predicted in zip(actual_sentiment, predicted_sentiment))

    recall = TP / (TP + FN) if TP + FN != 0 else 0.0
    return recall

def calculate_accuracy(actual_sentiment: pd.DataFrame, predicted_sentiment: pd.DataFrame):
    TP = sum((actual == predicted) for actual, predicted in zip(actual_sentiment, predicted_sentiment))
    total_samples = len(actual_sentiment)
    accuracy = TP / total_samples
    return accuracy

#Printing Confusion Matrix, Accurancy, Precision, Recall, F-measure
def calculate_metrics(size = 3):
    df = pd.read_csv(f'./sentiment_results{size}.csv')

    actual_sentiment = df['actual_sentiment']
    predicted_sentiment = df['predicted_sentiment']

    print(f"METRICS FOR {size} CLASSES", end = "\n\n")

    #Confusion matrix
    if size == 3:
        unique_labels = [-1, 0, 1]
        confusion_matrix = np.zeros((3, 3), dtype=int)
    elif size == 2:
        unique_labels = [-1, 1]
        confusion_matrix = np.zeros((2, 2), dtype=int)

    for actual, predicted in zip(actual_sentiment, predicted_sentiment):
        if actual in unique_labels and predicted in unique_labels:
            confusion_matrix[unique_labels.index(actual)][unique_labels.index(predicted)] += 1

    print("Confusion Matrix:")
    print(confusion_matrix)

    #Precision, Recall, Accurancy, F-measure
    precision_per_class = {}
    recall_per_class = {}
    for label in unique_labels:
        precision = calculate_precision(actual_sentiment, predicted_sentiment, label)
        recall = calculate_recall(actual_sentiment, predicted_sentiment, label)
        precision_per_class[label] = precision
        recall_per_class[label] = recall

    print("Accurancy:")
    print(calculate_accuracy(actual_sentiment,predicted_sentiment))

    print("Precision per Class:")
    print(precision_per_class)
    print("Average Precision:")
    avg_precision = sum(precision_per_class.values())/size
    print(avg_precision)

    print("Recall per Class:")
    print(recall_per_class)
    print("Average Recall:")
    avg_recall = sum(recall_per_class.values()) / size
    print(avg_recall)

    print("Average F-measure:")
    print(2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if avg_precision + avg_recall != 0 else 0.0, end = "\n\n")

def main():
    #dataset from: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
    data = read_data('./data/Reviews.csv', 1000)

    #Creating tokens and taggs and analyzing seperate sentence
    #text = input('Put your sentence here: ') #input of your example to test
    text = data['Text'][50] #example from Amazon data
    tokens = tokenize(text)
    print("Tokens of given sentence:")
    print(tokens)
    print("Taggs of given sentence:")
    print(tagged(tokens))
    print("Sentiment analysis:")
    print(sia.polarity_scores(text))

    #Calculating sentiment scores with VADER
    sentence_scores = calculate_sentiment_scores(data)

    #Output of created data
    print(sentence_scores)

    #Visualization of results
    scatter_plot(sentence_scores)
    bar_plot(sentence_scores)
    grouped_bar_plot(sentence_scores)

    #Measures for negative, (neutral) and positive sentiment
    create_csv(sentence_scores, 2) #csv for positive/negative
    create_csv(sentence_scores, 3) #csv for positive/negative/neutral
    calculate_metrics(2) #2 classes positive/negative, 2x2 matrix
    calculate_metrics(3) #3 classes positive/negative/neutral, 3x3 matrix

if __name__ == "__main__":
    main()


