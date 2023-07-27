# Sentiment Analysis with VADER

This project performs sentiment analysis on a dataset using VADER (Valence Aware Dictionary and sEntiment Reasoner) from the Natural Language Toolkit (NLTK) library. The dataset used for analysis is the Amazon reviews dataset.

## Requirements

- Python 3.6 or higher
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- vader_lexicon

## Installation

1. Clone the repository to your local machine:

git clone https://github.com/your_username/sentiment_analysis.git


2. Install the required packages using pip:

pip install -r requirements.txt


3. Download the required NLTK data:

Uncomment the following lines in the main.py file (if needed):

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('vader_lexicon')

4. Run the main.py file:

python main.py

##Overview

The sentiment analysis is performed using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. The code includes functions to read the dataset, tokenize the text, perform sentiment analysis, calculate sentiment scores, visualize the results, and create a CSV file with the actual and predicted sentiments. It also calculates various metrics such as precision, recall, F-measure, and accuracy based on the predicted sentiments.

##Dataset

The Amazon reviews dataset is used for this sentiment analysis. The dataset can be found at the following link: Amazon Reviews Dataset [https://www.kaggle.com/datasets/bittlingmayer/amazonreviews]

##Results

The code allows working with individual sentences or a specific database.

For the entered text, it creates tokens and tags, as well as performs sentiment analysis.

For more extensive databases the code generates various visualizations, including scatter plots and bar plots, to visualize the sentiment analysis results.

It also creates CSV files with the actual and predicted sentiments for both 2-class (positive/negative) and 3-class (positive/negative/neutral) sentiment categories. Metrics such as precision, recall, F-measure, and accuracy are calculated and printed for each sentiment category.

##Conclusion

This project demonstrates the application of VADER sentiment analysis on the Amazon reviews dataset. It provides insights into the sentiment distribution and allows for the evaluation of the sentiment analysis performance using various metrics.

