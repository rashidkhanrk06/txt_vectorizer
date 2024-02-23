
# Vectorizer Module

This Python module provides two classes for text vectorization - `CountVectorizer` and `TFIDFVectorizer`. These classes can be used to convert a collection of text documents into matrices of token counts or TF-IDF features.

## CountVectorizer

### Usage

```python
from vectorizer import CountVectorizer

vectorizer = CountVectorizer()
documents = ["This is a sample document.", "Another document for testing.", "Sample document with words."]
matrix = vectorizer.fit_transform(documents)
```

### Methods

#### `fit(documents)`

Learns the vocabulary from the given list of documents.

#### `transform(documents)`

Transforms the input documents into a sparse matrix of token counts.

#### `fit_transform(documents)`

Fits the model to the documents and transforms them in one step.

## TFIDFVectorizer

### Usage

```python
from vectorizer import TFIDFVectorizer

vectorizer = TFIDFVectorizer()
corpus = ["This is a sample document.", "Another document for testing.", "Sample document with words."]
matrix = vectorizer.fit_transform(corpus)
```

### Methods

#### `fit(corpus)`

Learns the vocabulary and term frequencies from the given corpus.

#### `transform(documents)`

Transforms the input documents into a TF-IDF matrix.

#### `fit_transform(corpus)`

Fits the model to the corpus and transforms it into a TF-IDF matrix in one step.

## Example

```python
# Using CountVectorizer
from vectorizer import CountVectorizer

vectorizer = CountVectorizer()
documents = ["This is a sample document.", "Another document for testing.", "Sample document with words."]
matrix = vectorizer.fit_transform(documents)

# Using TFIDFVectorizer
from vectorizer import TFIDFVectorizer

vectorizer = TFIDFVectorizer()
corpus = ["This is a sample document.", "Another document for testing.", "Sample document with words."]
matrix = vectorizer.fit_transform(corpus)
```

Feel free to modify and enhance this readme according to your preferences and any additional information you want to provide about the module.