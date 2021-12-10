# %%
__author__ = "Islem Maboud"
__aka__ = "CoderOne"
__github__ = "https://github.com/ipenywis"
__link__ = "https://www.youtube.com/c/coderone"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Islem Maboud"
__email__ = "islempenywis@gmail.com"


import numpy as np

import pandas as pd

#This will be used to use GoogleNews vectors
from gensim import models

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Reshape, Flatten, concatenate, Input

from keras.models import Model

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

from matplotlib import pyplot

import re

from os.path import isfile, join

import collections
import re
import string
import os
import nltk

# %% [markdown]
# ## Download Required Packages and Modes

# %%
## Downloading Required Packages and Models
#GOOGLE_NEWS_PATH = "./googleNews-word2vec"
GOOGLE_NEW_BIN_PATH = "./GoogleNews-vectors-negative300.bin"

#Github file is corrupted!
#!git clone "git@github.com:mmihaltz/word2vec-GoogleNews-vectors.git" $GOOGLE_NEWS_PATH

#!wget -P $GOOGLE_NEW_BIN_PATH -c -nc "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

# %% [markdown]
# ## Downloading Required Nltk processing packages

# %%
nltk.download('punkt')
nltk.download("stopwords")

# %% [markdown]
# ## Reading and Processing Data 

# %%
# Reading tweets from text file
datasetPath = './data/Tweets.txt'
tweetsData = pd.read_csv(datasetPath, header = None, delimiter='\t', encoding='utf-8-sig')
tweetsData.columns = ['Text', 'Label']
data = []

tweetsData.head()

# %% [markdown]
# ## Adding labels as integers instead of strings

# %%
tweetsData.Label.unique()


# %%
tweetsData.shape

# %%
pos = []
neg = []
obj = []
neutral = []
for l in tweetsData.Label:
    if l == 'OBJ':
        neg.append(0)
        pos.append(0)
        neutral.append(0)
        obj.append(1)
    elif l == 'NEUTRAL':
        neg.append(0)
        pos.append(0)
        neutral.append(1)
        obj.append(0)
    elif l == 'POS':
        neg.append(0)
        pos.append(1)
        neutral.append(0)
        obj.append(0)
    elif l == "NEG":
        neg.append(1)
        pos.append(0)
        neutral.append(0)
        obj.append(0)

tweetsData['Pos'] = pos
tweetsData['Neg'] = neg
tweetsData['Obj'] = obj
tweetsData['Neutral'] = neutral

tweetsData.head()

# %% [markdown]
# ## Cleaning Data

# %%
def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct

tweetsData['Text_Clean'] = tweetsData['Text'].apply(lambda x: remove_punct(x))
tweetsData.head()

# %%
# Tokenizing

from nltk import word_tokenize, WordNetLemmatizer
tokens = [word_tokenize(sen) for sen in tweetsData.Text_Clean]

print(tokens[0:8])

# %%
#Arabic Stopwords removal

from nltk.corpus import stopwords
stoplist = stopwords.words('arabic')

def remove_stop_words(tokens):
    return [word for word in tokens if word not in stoplist]

filtered_words = [remove_stop_words(sen) for sen in tokens] 
result = [' '.join(sen) for sen in filtered_words]

tweetsData['Text_Final'] = result
tweetsData['tokens'] = filtered_words

#Leave only the needed data (remove the rest)
tweetsData = tweetsData[['Text_Final', 'tokens', 'Label', 'Pos', 'Neg', 'Obj', 'Neutral']]

#Is it really removed? let's see
tweetsData[:5]

# %%
# Check the removed stopwords from tokens too
print(tweetsData.tokens[0:8])

# %% [markdown]
# ## Splitting Data into Train and Test

# %%
data_train, data_test = train_test_split(tweetsData, test_size=0.10, random_state=42)

# %%
#Calculating the data size for training
all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
#What's the size of the vocabulary for train? 
TRAINING_VOCAB = sorted(list(set(all_training_words)))

#Verbose info about our split datasets train
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))

# %%
#Calculating the data size for testing
all_test_words = [word for tokens in data_test["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test["tokens"]]
#Again! What's the size of the vocabulary for test? 
TEST_VOCAB = sorted(list(set(all_test_words)))

#Verbose info about our split datasets test
print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
print("Max sentence length is %s" % max(test_sentence_lengths))

# %% [markdown]
# ## Use Google News Word2Vec Model to create embeddings and have better results
# 
#   The idea behind this is using the word2vec model to create embeddings for each word in the dataset. This will help us to have better results when we try to predict the sentiment of a tweet. 

# %%
#Load google news bin
print(GOOGLE_NEW_BIN_PATH)
word2vec = models.KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True)

# %%
# Functions for manipulating and getting the embeddings 
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

# %% [markdown]
# ## Getting the actuall embeddings to the training dataset

# %%
training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)

MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300

# %% [markdown]
# ## Next step we are going to Pad sequences and toknize them

# %%
# Using the Tokenizer to convert words to integers
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train["Text_Final"].tolist())
#Finally extract the sequences from the generated tokens of the training data
training_sequences = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))

# %%
# Extract the training data as padded sequences
train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# %%
# Get training data embedding weihgts
train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)

#How does the shape look like?
print(train_embedding_weights.shape)

# %%
#Let's do the same for the test data
test_sequences = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_cnn_data.size


# %% [markdown]
# # Using RNN-LSTM

# %%
label_names = ['Pos', 'Neg', 'Obj', 'Neutral']

y_train = data_train[label_names].values

x_train = train_cnn_data
y_tr = y_train

# %% [markdown]
# ## Definning Our CNN-LSTM Model

# %%
# The reason we did a function is we may want to use it later without replicating the same code
# The function will create all the model layers and return the model at the end
def recurrent_nn(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=False)
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

#     lstm = LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedded_sequences)
    lstm = LSTM(256)(embedded_sequences)
    
    x = Dense(128, activation='relu')(lstm)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model

# %%
# Creating our Model
model = recurrent_nn(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                len(list(label_names)))

# %% [markdown]
# ## Training the RNN-LSTM Model

# %%
num_epochs = 5
batch_size = 256

train_hist = model.fit(x_train, y_tr, epochs=num_epochs, validation_split=0.1, shuffle=True, batch_size=batch_size)

# %% [markdown]
# ## Testing the Trained RNN-LSTM Model

# %%
predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)
labels = ["POS", "NEG", "OBJ", "NEUTRAL"]

predictions.shape

prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])

# %% [markdown]
# ## What's the accuracy of the model when running against the test data

# %%
sum(data_test.Label==prediction_labels)/len(prediction_labels)

# %% [markdown]
# ## Ploting the ROC for Accuracy for Train and Test

# %%
import matplotlib.pyplot as plt


# list all data in history
print(train_hist.history.keys())
# summarize history for accuracy
plt.plot(train_hist.history['acc'])
plt.plot(train_hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(train_hist.history['loss'])
plt.plot(train_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


