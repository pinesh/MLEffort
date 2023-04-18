# code for Glove word embedding
import tensorflow as tf
from tensorflow.python.keras.engine

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import collections
import pathlib

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import losses
from keras import utils
from keras.layers import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text

x = {'text', 'the', 'leader', 'prime',
     'natural', 'language'}

# create the dict.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)


# number of unique words in dict.
print("Number of unique words in dictionary=",
      len(tokenizer.word_index))
print("Dictionary is = ", tokenizer.word_index)


# download glove and unzip it in Notebook.
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove*.zip

# vocab: 'the': 1, mapping of words with
# integers in seq. 1,2,3..
# embedding: 1->dense vector
def embedding_for_vocab(filepath, word_index,
                        embedding_dim):
    vocab_size = len(word_index) + 1

    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size,
                                       embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix_vocab


# matrix for vocab: word_index
embedding_dim = 50
embedding_matrix_vocab = embedding_for_vocab(
    '../glove.6B.50d.txt', tokenizer.word_index,
    embedding_dim)

print("Dense vector for first word is => ",
      embedding_matrix_vocab[1])



"""
Memo to self, 

for types, 
B-product(H/O/A)
I-proidct(H/O/A)

Product "For" has capital, for lowercase is not. 

the steps will be to perform the 4 types of word vector one inccludes a cnn layer, concat output with rest

then the network consists of
concat
drop
keras BLSTM
dropout
CRF keras

Done


Word Level Embeddings.
<-assign each word to glove, do not train
<- missing words are assigned a default random

check word shape features



"""



