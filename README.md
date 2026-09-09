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
| `day1/` | `day1_tfidf.ipynb` and `news_dataset.csv` -- traditional NLP preprocessing (stopwords, lemmatization, n-grams) and a TF-IDF text classification model |

## Day 1: Traditional NLP Basics

`day1/day1_tfidf.ipynb` builds up text normalization, tokenization,
stopword removal, stemming vs. lemmatization, and n-grams by hand, then
bundles it all into one preprocessing function applied across the dataset.
The actual new material is TF-IDF (`TfidfVectorizer`) -- weighting words by
how distinctive they are across the dataset, not just how often they
appear, unlike `CountVectorizer`'s plain word counts used since Week 2.

TF-IDF + Logistic Regression got 62.5% accuracy. A controlled comparison
(same cleaned text, same split, same model, only the vectorizer swapped)
found `CountVectorizer` actually did *better* -- 71.4% -- the opposite of
the common assumption that TF-IDF always wins. Likely explanation: IDF
weighting is estimated from only 222 training documents, a small enough
corpus that the "how common is this word" estimate is noisy, and some
words TF-IDF downweighted for being frequent (like "stock") may actually
be strong, reliable signals within this specific dataset. Same theme as
Week 2's XGBoost-loses-to-Gradient-Boosting and
RandomizedSearchCV-picks-a-worse-model findings: a more sophisticated
technique doesn't automatically win, especially on small data.
