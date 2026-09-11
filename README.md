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
| `day2/` | `day2_embeddings.ipynb` and `news_dataset.csv` -- pretrained GloVe word embeddings, semantic similarity, sentence embeddings, and IDF-weighted averaging |
| `day3/` | `day3_transformer_notes.md` -- conceptual notes walking through how a transformer processes one real headline (self-attention, Q/K/V, positional encoding, multi-head, encoder/decoder) |
| `day4/` | `day4_bert.ipynb` and `news_dataset.csv` -- fine-tuning pretrained DistilBERT for the 6-way headline classification (the saved model is ~268MB so it's gitignored, it regenerates when the notebook runs) |
| `day5/` | `day5_ner_pipeline.ipynb` and `news_dataset.csv` -- NER on our headlines, plus a QA demo, a summarization demo, and a raw BIO tagging demo |

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

## Day 2: Word Embeddings and Semantic Text Representation

`day2/day2_embeddings.ipynb` uses pretrained GloVe vectors
(`glove-wiki-gigaword-100`, 400K words, 100 dimensions each, trained by
Stanford on Wikipedia + Gigaword) instead of training embeddings from
scratch -- 278 rows isn't nearly enough data for that. Semantic similarity
checks out immediately: "stock" is 85% similar to "shares", only 16%
similar to "banana". Headlines get represented as sentence embeddings by
averaging their words' vectors together, then classified the same way as
Day 1.

Results kept getting worse, not better, as more sophistication got added:

| Approach | Accuracy |
| --- | --- |
| CountVectorizer (Day 1) | 71.4% |
| TF-IDF (Day 1) | 62.5% |
| Plain averaged GloVe | 58.9% |
| IDF-weighted GloVe | 50.0% |

IDF-weighted averaging (weighting each word's vector by how rare it is
before averaging) made things *worse* than plain averaging -- the same
exact pattern as Day 1's TF-IDF-loses-to-CountVectorizer finding, now
showing up a second time. IDF estimates from only 222 documents are
genuinely noisy on a corpus this small, and "rare" doesn't reliably mean
"meaningful" here. Pretrained embeddings captured real semantic
relationships convincingly, but that didn't translate into a better
classifier on this small, narrow dataset -- simple word-presence signals
keep winning.

## Day 3: Transformer Architecture

`day3/day3_transformer_notes.md` is a written walkthrough (the deliverable
is "Transformer notes", not code) tracing how a transformer processes one
real headline from the dataset -- "Why Salesforce Stock Rallied Today" --
through every stage: tokenization and embeddings, positional encoding (a
patch for the fact that self-attention throws away word order),
self-attention via Query/Key/Value, multi-head attention, the residual +
feed-forward transformer block, and how encoders (BERT-style) differ from
decoders (GPT-style). The core shift from Week 3's LSTM: no sequential
chain and no fading memory -- every word gets direct parallel access to
every other word in one step.

## Day 4: BERT and RoBERTa for Text Classification

`day4/day4_bert.ipynb` fine-tunes pretrained DistilBERT for the 6-way
headline classification. It covers input IDs, attention masks, subword
tokenization ("China's" splits into three tokens), and what the
UNEXPECTED/MISSING load report means (the pretrained masked-language-
modeling head gets discarded, a fresh random classification head gets
created and trained). The standard fine-tuning recipe (lr 2e-5, 4 epochs)
only got 57% with the loss still dropping -- that recipe assumes a bigger
dataset, so it was bumped to lr 3e-5 for 15 epochs, landing at 69.6%.

| Approach | Accuracy |
| --- | --- |
| CountVectorizer (Day 1) | 71.4% |
| DistilBERT fine-tuned | 69.6% |
| TF-IDF (Day 1) | 62.5% |
| Plain averaged GloVe (Day 2) | 58.9% |
| Week 3 LSTM | 50% |
| Week 3 ANN from scratch | 39-52% |

This is the first deep learning model in the whole project that actually
competes with the classical methods -- it basically matches CountVectorizer
and beats everything else deep-learning. It's also the only model all
project that predicted any Health headlines right. The reason it works
where Week 3 failed is the pretraining: Week 3's models had to learn
English and the task at once from 222 rows (impossible), while DistilBERT
already knew English and only needed to learn which words point to which
category. Fine-tuning a pretrained model is the right tool for a small
dataset; training one from scratch is not.

## Day 5: NER, QA, and Summarization

`day5/day5_ner_pipeline.ipynb` picked NER as the main build since our
headlines are full of entities and it's the most useful of the three in
real work. added a small QA demo and a small summarization demo too since
they fit in the same notebook, plus a raw BIO tagging demo since the main
NER step hides the raw tags.

three separate pipeline shortcuts (question-answering, summarization, and
grouped_entities for NER) are all gone in this transformers version. had
to rebuild each one directly with the underlying model class instead of
the convenience wrapper. annoying at first but it forced seeing what each
task actually does mechanically instead of hiding behind a shortcut.

NER on our own headlines is a mixed bag. clean on generic names and
companies (Elon Musk, AMD, Nvidia all near 100% confidence), but falls
apart on financial specific stuff, S&P 500 got split weirdly, CoreWeave
came back in two broken pieces, ETF got chopped down with lower
confidence. same story as Day 4, this model was trained on old general
news, not finance, so it's strong in general but needs fine-tuning to
really work on a specific domain.

BIO tagging turned up something not in the textbook version either. every
tag came back as I- with zero B- tags anywhere, even on the very first
piece of an entity. turns out that's a real quirk of this specific model,
not a bug, the original CoNLL-2003 scheme barely ever needs a B- tag so
the model learned it can skip it entirely.
