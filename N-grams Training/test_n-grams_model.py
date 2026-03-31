"""
This script can be used to test the accuracy of the N-grams model.
Please edit the script to suit your testing needs.
"""

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from nltk.tokenize import RegexpTokenizer
import torch
import torch.nn as nn
import pickle

ngram = 10

with open("Models/vocab.pkl", "rb") as f:
  vocab = pickle.load(f)
with open(f"Models/X_test_{ngram}-gram.pkl", "rb") as f:
  X_test = pickle.load(f)
with open(f"Models/y_test_{ngram}-gram.pkl", "rb") as f:
  y_test = pickle.load(f)


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
with open(f"Models/trained_model_{ngram}-gram.pkl", "rb") as f:
  word_predictor_model = pickle.load(f)



def predict_next_word(context_words):
  context_idxs = torch.tensor([[word_to_idx[w] for w in context_words]], device=device_gpu)
  word_predictor_model.eval()
  with torch.no_grad():
    logits = word_predictor_model(context_idxs).squeeze(0)
    pred_ix = torch.argmax(logits, dim=0).item()
  return idx_to_word[pred_ix]

def predict_top_k_words(context_words, k=10, temperature=1.0):
  context_idxs = torch.tensor([[word_to_idx[w] for w in context_words]], device=device_gpu)
  word_predictor_model.eval()
  with torch.no_grad():
    logits = word_predictor_model(context_idxs).squeeze(0) / temperature
    probabilities = torch.softmax(logits, dim=0)

    top_probabilities, top_indices = torch.topk(probabilities, k)

  top_probabilities = top_probabilities.cpu()
  top_indices = top_indices.cpu()
  # return [(idx_to_word[i.item()]) for i in top_indices]
  return [(idx_to_word[i.item()], p.item()) for i, p in zip(top_indices, top_probabilities)]



stop_word_list = ["the","of","and","to","in","a","is","that","for","are","as","with","it","on","be","or","by","then", "it's"]

# correct_count = 0
# total_count = 0
# for i in range(len(y_test)):
#   # if (y_test[i] in stop_word_list): continue
#   if (y_test[i] == predict_next_word(X_test[i])):
#     correct_count += 1
#   total_count += 1
#   if (total_count > 99999): break
# accuracy = correct_count / total_count
# print(f"accuracy_1w: {accuracy*100, total_count}")

# correct_count = 0
# total_count = 0
# for i in range(len(y_test)):
#   if (y_test[i] in stop_word_list): continue
#   if (y_test[i] == predict_next_word(X_test[i])):
#     correct_count += 1
#   total_count += 1
#   if (total_count > 99999): break
# accuracy = correct_count / total_count
# print(f"accuracy_1w*: {accuracy*100, total_count}")

# correct_count = 0
# total_count = 0
# for i in range(len(y_test)):
#   # if (y_test[i] in stop_word_list): continue
#   if (y_test[i] in predict_top_k_words(X_test[i], 10)):
#     correct_count += 1
#   total_count += 1
#   if (total_count > 99999): break
# accuracy = correct_count / total_count
# print(f"accuracy_3w: {accuracy*100, total_count}")

# correct_count = 0
# total_count = 0
# for i in range(len(y_test)):
#   if (y_test[i] in stop_word_list): continue
#   if (y_test[i] in predict_top_k_words(X_test[i], 10)):
#     correct_count += 1
#   total_count += 1
#   if (total_count > 99999): break
# accuracy = correct_count / total_count
# print(f"accuracy_3w*: {accuracy*100, total_count}")



while True:
  usr_inp = input(f"Enter {ngram} words: ")
  if usr_inp == "exit": break
  tokenizer = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?")
  inp_list = tokenizer.tokenize(usr_inp)
  if (len(inp_list) == ngram):
    # print("next word: ", predict_next_word(inp_list))
    print("next word: ", predict_top_k_words(inp_list, 50, 1))
  else: continue
