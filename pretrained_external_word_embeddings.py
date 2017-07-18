'''This script loads pre-trained word embeddings (GloVe embeddings)
but keeps them external to the Keras models. Embeddings are used to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, Embedding, Masking, Conv2D, MaxPooling2D
from keras.models import Model
from pdb import set_trace as bp

np.random.seed(41)

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')


def get_random_vec(embedding_dims):
    return np.asarray(np.random.normal(0, 1, embedding_dims), dtype='float32')


def pad_word_array(word_array, MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre'):
    """Return a word array that is of a length MAX_SEQUENCE_LENGTH by truncating the original array or padding it

    Args:
        word_array:
        MAX_SEQUENCE_LENGTH:
        padding:
        truncating:

    Returns:

    """
    if len(word_array) > MAX_SEQUENCE_LENGTH:
        word_array = word_array[:MAX_SEQUENCE_LENGTH] if truncating == 'pre' else word_array[len(
            word_array) - MAX_SEQUENCE_LENGTH:]
    else:
        if padding == 'pre':
            word_array = word_array + ['<pad>'] * (MAX_SEQUENCE_LENGTH - len(word_array))
        else:
            word_array = ['<pad>'] * (MAX_SEQUENCE_LENGTH - len(word_array)) + word_array
    return word_array


embeddings_index = {}

f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))

for ind, line in enumerate(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    # if ind>1000:
    # break

# add the padding token vector
embeddings_index['<pad>'] = get_random_vec(EMBEDDING_DIM)
embeddings_index['<unk>'] = get_random_vec(EMBEDDING_DIM)

f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
texts_vectors = []  # list of vectors for each sentence
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]

                # convert the text segment to a list of words
                word_array = text_to_word_sequence(t)
                # truncate or pad the list of words with the <pad> token to get it to MAX_SEQUENCE_LENGTH
                word_array = pad_word_array(word_array, MAX_SEQUENCE_LENGTH)
                # get the vectors for these words
                sentence_matrix = [(embeddings_index[word] if word in embeddings_index else embeddings_index['<unk>'])
                                   for word in word_array]
                texts_vectors.append(sentence_matrix)
                f.close()
                labels.append(label_id)
                if len(texts_vectors) > 2000:
                    break

print('Found %s texts.' % len(texts_vectors))

data = np.asarray(texts_vectors)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

print('Training model.')

# Notice how the embedding layer is removed and the each input to the model is an embedding matrix representing the sequence
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dtype='float32')

x = Conv1D(128, 5, activation='relu')(sequence_input)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print(model.summary())
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))
