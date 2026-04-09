"""
This script calculates the speed changes of an audio file given a transcript.
It also computes the force alignment so speed changes can be mapped to timestamps.
It requires the correct models are present in Models/
"""


import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sentence_transformers import SentenceTransformer
from forcealign import ForceAlign
import re
import torch
import torch.nn as nn
import pickle


NAME = "name-of-file"
transcript_path = f"Input/{NAME}.txt"
audio_path = f"Input/{NAME}.wav"


# measuring sentence similarity and wordcounts per sentence #

model_directory = "Model_Data/sentence_similarity"

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# model.save_pretrained(model_directory)

model = SentenceTransformer(model_directory)

f = open(transcript_path, encoding='utf-8')
transcript = f.read()
f.close()
transcript = re.sub('’', "'", transcript)

WORD = r"[a-z]+(?:'[a-z]+)?"  # this ignores nummeric numbers since ForceAlign does

# tokenizer = RegexpTokenizer(r"\b[\w']+\b")
tokenizer = RegexpTokenizer(WORD)

# remove_timestamps = re.sub(r"([0-9]{2}:){2}[0-9]{2}.[0-9]{3}", "", transcript)
remove_timestamps = re.sub(r"\[(.*?)\]", "", transcript.lower())
remove_newlines = re.sub(r"\n", "", remove_timestamps)
remove_parenthesis = re.sub(r"\([^)]*\)", "", remove_newlines)

clean_transcript = ' '.join(tokenizer.tokenize(remove_parenthesis))
tokenized_transcript = tokenizer.tokenize(clean_transcript)
sentences = np.array(re.split(r"[.?!] *", remove_parenthesis), dtype=str)

word_count = len(tokenized_transcript)

# perform force alignment #
alignment = ForceAlign(audio_file=audio_path, transcript=clean_transcript)
word_times = alignment.inference()

# # calculating times when no speech is present #
# silence_threshold = 0.500
# prev_word_end_time = word_times[1].time_start
# silent_sections = []
# for word in word_times:
#     if word.time_start > prev_word_end_time + silence_threshold:
#       silent_sections.append((prev_word_end_time+silence_threshold/2, word.time_start-silence_threshold/2))
#     prev_word_end_time = word.time_end
# total_silent_time = sum(map(lambda x: x[0]-x[1], silent_sections))

# get sentence word counts
sentence_word_counts = []
for i in sentences:
  # wc = len(re.findall(r"\b[\w']+\b", i))
  wc = len(re.findall(WORD, i))
  sentence_word_counts.append(wc)

# embed sentences into the huggingface embedding
embeddings = np.array(model.encode(sentences))

def measure_similarity(x, n):
  score = 0
  for i in range(1, n):
    score += np.dot(embeddings[x], embeddings[x-i]) / 2**i
    score += np.dot(embeddings[x], embeddings[x+i]) / 2**i
  return score * (2**n / (2 * 2**n - 1))

sentence_similarities = []
N = 5
min_x = N
max_x = len(embeddings)-N
for i in range(N):
  sentence_similarities += [0.0] * i
for i in range(min_x, max_x):
  sentence_similarities += [measure_similarity(i, N)] * sentence_word_counts[i]
for i in range(len(tokenized_transcript) - max_x):
  sentence_similarities += [0.0] * i


# measuring word probabilities #
ngram = 10
with open("Model_Data/vocab.pkl", "rb") as f:
  vocab = pickle.load(f)
with open(f"Model_Data/X_test_{ngram}-gram.pkl", "rb") as f:
  X_test = pickle.load(f)
with open(f"Model_Data/y_test_{ngram}-gram.pkl", "rb") as f:
  y_test = pickle.load(f)


word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_count = len(vocab)
device_cpu = torch.device("cpu")


class NGramModel(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
    super().__init__()
    self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

    self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    self.fc = nn.Linear(hidden_dim, vocab_size)

  def forward(self, x):
    embeds = self.embeddings(x)
    lstm_out, _ = self.lstm(embeds)
    last_hidden = lstm_out[:, -1, :]
    out = self.fc(last_hidden)
    return out
  
word_predictor_model = NGramModel(
  vocab_size=vocab_count,
  embed_dim=300,
  hidden_dim=128,
  num_layers=1
)
with open(f"Model_Data/trained_model_{ngram}-gram.pkl", "rb") as f:
  word_predictor_model = pickle.load(f)
word_predictor_model.to(device_cpu)

def predict_top_k_words(context_words, k=10, temperature=1.0):
  for w in context_words:
    if w not in word_to_idx:
      return []
  context_idxs = torch.tensor([[word_to_idx[w] for w in context_words]], device=device_cpu)
  word_predictor_model.eval()
  with torch.no_grad():
    logits = word_predictor_model(context_idxs).squeeze(0) / temperature
    probabilities = torch.softmax(logits, dim=0)

    top_probabilities, top_indices = torch.topk(probabilities, k)

  top_probabilities = top_probabilities.cpu()
  top_indices = top_indices.cpu()
  # return [(idx_to_word[i.item()]) for i in top_indices]
  return [(idx_to_word[i.item()], p.item()) for i, p in zip(top_indices, top_probabilities)]


# iterate over transcript and calculate word probabilities

# Word probabilities
word_probability_list = [(tokenized_transcript[0], 0.0), (tokenized_transcript[1], 0.0), (tokenized_transcript[2], 0.0), (tokenized_transcript[3], 0.0), (tokenized_transcript[4], 0.0), (tokenized_transcript[5], 0.0), (tokenized_transcript[6], 0.0), (tokenized_transcript[7], 0.0), (tokenized_transcript[8], 0.0)]
for i in range((ngram-1), len(tokenized_transcript)):
  context_words = tokenized_transcript[i-(ngram-1):i]
  next_word_guesses = predict_top_k_words(context_words=context_words, k=100, temperature=1.0)
  actual_next_word = tokenized_transcript[i]
  word_probability = 0.0
  for guess in next_word_guesses:
    if actual_next_word == guess[0]:
      word_probability = guess[1]
      break
  word_probability_list.append((actual_next_word, word_probability))


wordspeed_deltas = [ word_probability_list[i][1] + sentence_similarities[i] for i in range(word_count) ]


# save word speed and time data to be used in later script
with open(f"Cache/wordspeed_deltas{NAME}.pkl", "wb") as f:
  pickle.dump(wordspeed_deltas, f)
with open(f"Cache/word_times{NAME}.pkl", "wb") as f:
  pickle.dump(word_times, f)
