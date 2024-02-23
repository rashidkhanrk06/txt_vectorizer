import numpy as np
import pandas as pd

class CountVectorizer:
    """
    A simple Count Vectorizer class for converting a collection of text documents to a matrix of token counts.

    Attributes:
        word_index (dict): A dictionary mapping unique words to their corresponding indices.
        unique_words (list): A list containing unique words encountered during the fitting process.
        indexes (list): A list of indexes corresponding to the unique words.
        _n_sample (int): Number of samples in the input data.

    Methods:
        __init__(): Initializes an instance of CountVectorizer.
        fit(documents): Learns the vocabulary from the given list of documents.
        transform(documents): Transforms the input documents into a sparse matrix of token counts.
        fit_transform(documents): Fits the model to the documents and transforms them in one step.

    Example:
        vectorizer = CountVectorizer()
        documents = ["This is a sample document.", "Another document for testing.", "Sample document with words."]
        matrix = vectorizer.fit_transform(documents)
    """

    def __init__(self):
        """
        Initializes an instance of the CountVectorizer.

        Attributes:
            word_index (dict): A dictionary mapping unique words to their corresponding indices.
            unique_words (list): A list containing unique words encountered during the fitting process.
            indexes (list): A list of indexes corresponding to the unique words.
            _n_sample (int): Number of samples in the input data.
        """
        self.word_index = {}
        self.unique_words = []
        self.indexes = [int]
        self._n_sample = None

    def fit(self, documents):
        """
        Learns the vocabulary from the given list of documents.

        Parameters:
            documents (list): A list of text documents.

        Returns:
            None
        """
        self._n_sample = len(documents)

        for document in documents:
            words = document.split()
            for word in words:
                if word not in self.word_index:
                    self.word_index[word] = len(self.unique_words)
                    self.unique_words.append(word)
                    self.indexes.append(self.word_index[word])

    def transform(self, documents):
        """
        Transforms the input documents into a sparse matrix of token counts.

        Parameters:
            documents (list): A list of text documents.

        Returns:
            np.ndarray: A NumPy array representing the bag-of-words matrix.
        """
        bow_matrix = np.zeros((len(documents), len(self.unique_words)), dtype=int)
        for i, document in enumerate(documents):
            words = document.split()
            for word in words:
                if word in self.word_index:
                    bow_matrix[i, self.word_index[word]] += 1
        return bow_matrix

    def fit_transform(self, documents):
        """
        Fits the model to the documents and transforms them in one step.

        Parameters:
            documents (list): A list of text documents.

        Returns:
            np.ndarray: A NumPy array representing the bag-of-words matrix.
        """
        self.fit(documents)
        matrix = self.transform(documents)
        return matrix


import numpy as np

class TFIDFVectorizer:
    """
    A TF-IDF Vectorizer class for converting a collection of text documents to a matrix of TF-IDF features.

    Attributes:
        _tf (list): A list of dictionaries containing term frequencies for each document.
        term_in_doc (dict): A dictionary keeping track of the number of documents each term appears in.
        idf_ (dict): A dictionary containing the inverse document frequencies for each term.
        vocabulary_ (dict): A dictionary mapping terms to their corresponding indices in the vocabulary.
        documents (list): A list of input text documents.
        _n_samples (int): Number of samples in the input data.

    Methods:
        __init__(): Initializes an instance of TFIDFVectorizer.
        fit(corpus): Learns the vocabulary and term frequencies from the given corpus.
        transform(documents): Transforms the input documents into a TF-IDF matrix.
        fit_transform(corpus): Fits the model to the corpus and transforms it into a TF-IDF matrix in one step.

    Example:
        vectorizer = TFIDFVectorizer()
        corpus = ["This is a sample document.", "Another document for testing.", "Sample document with words."]
        matrix = vectorizer.fit_transform(corpus)
    """

    def __init__(self):
        """
        Initializes an instance of the TFIDFVectorizer.

        Attributes:
            _tf (list): A list of dictionaries containing term frequencies for each document.
            term_in_doc (dict): A dictionary keeping track of the number of documents each term appears in.
            idf_ (dict): A dictionary containing the inverse document frequencies for each term.
            vocabulary_ (dict): A dictionary mapping terms to their corresponding indices in the vocabulary.
            documents (list): A list of input text documents.
            _n_samples (int): Number of samples in the input data.
        """
        self._tf = []
        self.term_in_doc = {}
        self.idf_ = {}
        self.vocabulary_ = {}
        self.documents = []
        self._n_samples = None

    def fit(self, corpus):
        """
        Learns the vocabulary and term frequencies from the given corpus.

        Parameters:
            corpus (list): A list of text documents.

        Returns:
            None
        """
        self.documents = corpus
        self._n_samples = len(corpus)
        index = 0

        for document in corpus:
            words = document.split()
            terms, term_counts = np.unique(words, return_counts=True)
            self._tf.append(dict(zip(terms, term_counts / len(words))))
            for word in terms:
                if word not in self.term_in_doc:
                    self.term_in_doc[word] = 1
                    self.vocabulary_[word] = index
                    index += 1
                else:
                    self.term_in_doc[word] += 1

    def transform(self, documents):
        """
        Transforms the input documents into a TF-IDF matrix.

        Parameters:
            documents (list): A list of text documents.

        Returns:
            np.ndarray: A NumPy array representing the TF-IDF matrix.
        """
        tf_idf_matrix = np.zeros((self._n_samples, len(self.vocabulary_)), dtype=float)

        for index, document in enumerate(self.documents):
            words = document.split()
            for word in words:
                if word not in self.idf_:
                    self.idf_[word] = np.log((self._n_samples + 1) / (self.term_in_doc[word] + 1)) + 1
                tf_idf_matrix[index, self.vocabulary_[word]] = self._tf[index][word] * self.idf_[word]

        return tf_idf_matrix

    def fit_transform(self, corpus):
        """
        Fits the model to the corpus and transforms it into a TF-IDF matrix in one step.

        Parameters:
            corpus (list): A list of text documents.

        Returns:
            np.ndarray: A NumPy array representing the TF-IDF matrix.
        """
        self.fit(corpus)
        matrix = self.transform(corpus)
        return matrix
