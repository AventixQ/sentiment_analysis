#Amazon data set VADER analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('vader_lexicon')

plt.style.use('ggplot')#


#reading data and basic analysis
data = pd.read_csv('./data/Reviews.csv')
data = data.head(1000) #using 1000 rows
#print(data.shape)

ax = data['Score'].value_counts().sort_index().plot(kind = 'bar',title = 'Count of Reviews',\
                                               figsize = (10, 5))
ax.set_xlabel('Review Stars')
plt.show()

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

sentence_scores = sentence_scores.reset_index().rename(columns={'index':'Id'})
#print(sentence_scores)
sentence_scores = sentence_scores.merge(data, how = 'left')
#print(sentence_scores['compound'])
#print(sentence_scores.head())


#WYKRES 1
plt.scatter(sentence_scores['Score'], sentence_scores['compound'], color='red', marker='.')
plt.xlabel('Score')
plt.ylabel('Compound')
plt.title('VADER results by Score')
plt.grid(True)
plt.tight_layout()
plt.show()

#WYKRES 2
sentiment_means = sentence_scores.groupby('Score')['compound'].mean()
stars = sentiment_means.index
plt.bar(stars, sentiment_means, align='center', alpha=0.7)
plt.xlabel('Score')
plt.ylabel('Average sentiment result')
plt.title('Distributions of VADER results by Star Review')
plt.grid(True)
plt.tight_layout()
plt.show()

#WYKRES 3
fig, axs = plt.subplots(2,2,figsize=(10, 10))
sns.barplot(data = sentence_scores, x = sentence_scores['Score'], y = sentence_scores['compound'], ax=axs[0, 0])
sns.barplot(data = sentence_scores, x = sentence_scores['Score'], y = sentence_scores['pos'], ax=axs[0, 1])
sns.barplot(data = sentence_scores, x = sentence_scores['Score'], y = sentence_scores['neu'], ax=axs[1, 0])
sns.barplot(data = sentence_scores, x = sentence_scores['Score'], y = sentence_scores['neg'], ax=axs[1, 1])
axs[0, 0].set_title("Compound")
axs[0, 1].set_title("Positive")
axs[1, 0].set_title("Neutral")
axs[1, 1].set_title("Negative")
plt.show()

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



