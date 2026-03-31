"""
Please use this script to generate your training data.
ted_talks_en.csv can be found at https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset/data.
"""

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import pandas as pd
# np.set_printoptions(threshold=np.inf)
import re
from nltk.tokenize import RegexpTokenizer
import random
import pickle


# reading data from csv
dataframe = pd.read_csv(
  "Datasets/ted_talks_en.csv",
  header=0,
  usecols=['transcript']
  )
raw_transcript_list = dataframe['transcript'].to_list()
transcript_count = len(raw_transcript_list)

# adding words.txt to vocab set
# with open("Datasets/words.txt") as f:
#   vocab = set(line.strip() for line in f)
vocab = set()


# cleaning input data
tokenizer = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?")
tokenized_transcripts = []
for t in raw_transcript_list:
  t = t.lower()
  t = re.sub(r"\([^)]*\)", "", t)     # removing parenthesized audience actions
  tokens = tokenizer.tokenize(t)
  tokenized_transcripts.append(tokens)


# vocabulary to index mappings
for t in tokenized_transcripts:
  for i in t:
    vocab.add(i)
vocab = sorted(vocab)
with open("Models/vocab.pkl", "wb") as f:
  pickle.dump(vocab, f)

# creating X and y data (full)
X_str = []
y_str = []
ngram = 10
for ts in tokenized_transcripts:
  for i in range(len(ts) - ngram):
    context = ts[i:i+ngram]
    content = ts[i+ngram]

    X_str.append(context)
    y_str.append(content)


# creating train/validate/test split (70/15/15)
total_samples = len(y_str)
train_idx = round(total_samples*0.7)
validate_idx = train_idx + round(total_samples*0.15)
random.seed(67)
X_shuffled = random.sample(X_str, total_samples)
random.seed(67)
y_shuffled = random.sample(y_str, total_samples)
X_train = X_shuffled[:train_idx]
y_train = y_shuffled[:train_idx]
X_validate = X_shuffled[train_idx:validate_idx]
y_validate = y_shuffled[train_idx:validate_idx]
X_test = X_shuffled[validate_idx:total_samples-1]
y_test = y_shuffled[validate_idx:total_samples-1]

# caching sample data
with open(f"Models/X_train_{ngram}-gram.pkl", "wb") as f: pickle.dump(X_train, f)
with open(f"Models/y_train_{ngram}-gram.pkl", "wb") as f: pickle.dump(y_train, f)
with open(f"Models/X_validate_{ngram}-gram.pkl", "wb") as f: pickle.dump(X_validate, f)
with open(f"Models/y_validate_{ngram}-gram.pkl", "wb") as f: pickle.dump(y_validate, f)
with open(f"Models/X_test_{ngram}-gram.pkl", "wb") as f: pickle.dump(X_test, f)
with open(f"Models/y_test_{ngram}-gram.pkl", "wb") as f: pickle.dump(y_test, f)
