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

model_directory = "Models/sentence_similarity"

if os.path.exists(model_directory):
  model = SentenceTransformer(model_directory)
else:
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  model.save_pretrained(model_directory)

# read transcript from file
f = open(transcript_path, encoding='utf-8')
transcript = f.read()
f.close()


# remove_timestamps = re.sub(r"([0-9]{2}:){2}[0-9]{2}.[0-9]{3}", "", transcript)
remove_timestamps = re.sub(r"\[(.*?)\]", "", transcript)
remove_newlines = re.sub(r"\n", "", remove_timestamps)
remove_parenthesis = re.sub(r"\([^)]*\)", "", remove_newlines)
sentences = np.array(re.split(r"[.?!] *", remove_parenthesis), dtype=str)

# get sentence word counts
sentence_word_counts = []
for i in sentences:
  word_count = len(re.findall(r"\b[\w']+\b", i))
  sentence_word_counts.append(word_count)

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
for i in sentence_word_counts[:N]:
  sentence_similarities += [0.0] * i
for i in range(min_x, max_x):
  sentence_similarities += [measure_similarity(i, N)] * sentence_word_counts[i]


# get vocab data for encoding #
with open("Models/vocab.pkl", "rb") as f:
  vocab = pickle.load(f)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_count = len(vocab)
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
word_predictor_model.to(device_gpu)
# measuring word probabilities. This uses a previously trained n-grams model #
ngram = 10
with open(f"Models/trained_model_{ngram}-gram.pkl", "rb") as f:
  word_predictor_model = pickle.load(f)

def predict_top_k_words(context_words, k=10, temperature=1.0):
  for w in context_words:
    if w not in word_to_idx:
      return []
  context_idxs = torch.tensor([[word_to_idx[w] for w in context_words]], device=device_gpu)
  word_predictor_model.eval()
  with torch.no_grad():
    logits = word_predictor_model(context_idxs).squeeze(0) / temperature
    probabilities = torch.softmax(logits, dim=0)

    top_probabilities, top_indices = torch.topk(probabilities, k)

  top_probabilities = top_probabilities.cpu()
  top_indices = top_indices.cpu()
  return [(idx_to_word[i.item()], p.item()) for i, p in zip(top_indices, top_probabilities)]


# iterate over transcript and calculate word probabilities
tokenizer = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?")
clean_transcript = tokenizer.tokenize(re.sub(r"\([^)]*\)", "", transcript.lower()))
clean_transcript = ' '.join(clean_transcript)
tokenized_transcript = tokenizer.tokenize(re.sub(r"\([^)]*\)", "", transcript.lower()))

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


# perform force alignment and compute base speaking rate #
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

word_count = len(word_times)
total_speaking_time = word_times[word_count-1].time_start - word_times[0].time_end #- total_silent_time
base_speaking_rate_minutes = word_count * (60 / total_speaking_time)

base_speed_increase = 280 / base_speaking_rate_minutes
dynamic_speed_multiplier = 50 / base_speaking_rate_minutes
wordspeed_deltas = [ word_probability_list[i][1] + sentence_similarities[i] for i in range(len(sentence_similarities)) ]


# save word speed and time data to be used in later script
with open(f"Cache/wordspeed_deltas{NAME}.pkl", "wb") as f:
  pickle.dump(wordspeed_deltas, f)
with open(f"Cache/word_times{NAME}.pkl", "wb") as f:
  pickle.dump(word_times, f)
