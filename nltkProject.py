# Made using
# https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk

# THIS CODE PERFORMS SENTIMENT ANALYSIS OF TWEETS FROM NLTK LIBRARY AND CUSTOM TWEETS WRITTEN AS VARIABLE "CUSTOM_TWEET"

# downloads

# nltk.download('twitter_samples')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')


import re  # RegEx
import string
import random

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk import NaiveBayesClassifier, FreqDist, classify
from nltk import word_tokenize


# Removing noise + normalization + lemmatization
def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    # .sub() - replace pattern (http:// -> empty string)
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)

        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# Determining word density
def get_all_words(cleaned_token_list):
    for tokens in cleaned_token_list:
        for token in tokens:
            yield token

# Convert
def get_tweets_for_model(cleaned_token_list):
    for tweet_tokens in cleaned_token_list:
        yield dict([token, True] for token in tweet_tokens)


# ----------------------"MAIN"--------------------------------------

# Tokenizing data
# strings() -> print all tweets within a dataset as strings
positive_tweets = twitter_samples.strings('positive_tweets.json')  # 5000 negative sentiment Tweets
negative_tweets = twitter_samples.strings('negative_tweets.json')  # 5000 positive sentiment Tweets
text = twitter_samples.strings('tweets.20150430-223406.json')  # 20000 no sentiment Tweets

# Object tokenizing "positive_tweets" dataset
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')  # [0]

# for testing
# print(tweet_tokens[0])

stop_words = stopwords.words('english')

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

# cleaning tokens
for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

all_pos_words = get_all_words(positive_cleaned_tokens_list)

freq_dist_pos = FreqDist(all_pos_words)
print("Frequency distribution: ")
print(freq_dist_pos.most_common(10))

# preparing data
positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

# splitting dataset for training NaiveBayesClassifier
positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

# shuffle to avoid bias
random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

# building model
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))

# First review of game R6Siege off of Steam
custom_tweet = '''
It used to be a great game, but now ruined.

PVE: Terrorist Hunt renamed 'Training Mode', bombers removed, renown capped, xp capped, corpses disappear,
blood splatter reduced, and ai smart functionality reduced.

PVP: A costume freak-show, fast-paced run and gun, spray and pray shooter. It's no longer about objectives or
teamwork, it's about splitting up and running around like rabbits trying to get the most kills.

Made by Bugίsoft, also known as Ubίshίt. Their policy is to milk money off their customers.
Care, compassion, and ethics do not apply because at the very top sits a white devil,
also known as a contemporary barbarian.

Their games have various problems that needs to be fixed, but they don't care.
'''

custom_token = remove_noise(word_tokenize(custom_tweet))


print(custom_tweet)
print("\nCustom tweet sentiment: ")
print(classifier.classify(dict([token, True] for token in custom_token)))
