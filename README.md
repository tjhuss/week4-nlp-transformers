# NLP, Transformers, BERT, RoBERTa, T5, and LLM Architecture

Week 4 of a self-directed AI/ML/DL internship prep program. Moves from
Week 3's deep learning mechanics (neurons, activations, RNN/LSTM) into
NLP techniques and transformer-based architecture -- TF-IDF, word
embeddings, self-attention, and eventually BERT/RoBERTa and LLM
fundamentals.

## Setup

```
pip install -r requirements.txt
```

Also needs a few nltk data packages (stopwords, wordnet, punkt) --
downloaded once via:

```
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"
```

## Project structure

| Folder | Contents |
| --- | --- |
| `day1/` | `day1_tfidf.py` and `news_dataset.csv` -- traditional NLP preprocessing (stopwords, lemmatization, n-grams) and a TF-IDF text classification model |
