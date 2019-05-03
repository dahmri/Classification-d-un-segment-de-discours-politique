import numpy as np
import re
import itertools
from collections import Counter
from keras.preprocessing import sequence


def clean(string):
    """
    Cleaning sentences
    """

    string = re.sub(r"[^A-Za-z'àâäçéèêëîïôöùûüÿ]", " ", string)
    string = re.sub(r"\'", "\' ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_sentences_and_labels():
    """
    Loading Data from text files and creating labels
    """

    print ("load_data_and_labels")
    # Load data from files
    chirac_txt = list(open("./Data/chirac.txt",encoding='latin-1').readlines())
    mitterrand_txt = list(open("./Data/mitterrand.txt",encoding='latin-1').readlines())

    #Cleaning and Splitting by words
    sentences = chirac_txt + mitterrand_txt
    sentences = [clean(s) for s in sentences]
    sentences = [s.split() for s in sentences]

    #Creating the lables
    labels=[0]*len(sentences)
    y=np.array(labels)
    y[len(chirac_txt):]=1
    #Label 0 for chirac
    #Label 1 for mitterrand

    return sentences, y


def pad_sequences(sentences, padding_word):
    """
    Padding sentences, all sentences will have the same length
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sequences = sequence.pad_sequences(sentences, maxlen=sequence_length, padding="post", truncating="post",value=padding_word)
    return padded_sequences


def build_vocabulary(sentences):
    """
    Building a dictionary of vocabulary from Data
    """

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Mapping from index to word
    liste_words = [x[0] for x in word_counts.most_common()]
    liste_words.append('PAD')

    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(liste_words)}
    return vocabulary


def build_sequences(sentences,vocabulary):
    """
    Building sequences from sentences
    """

    sequences = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return sequences


def load_data():
    """
    Loading the data and creating labels and vocabulary
    """

    sentences, labels = load_sentences_and_labels()
    vocabulary = build_vocabulary(sentences)
    sequences=build_sequences(sentences,vocabulary)
    data=pad_sequences(sequences,vocabulary['PAD'])

    return data, labels, vocabulary
