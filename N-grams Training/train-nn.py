"""
This script can be used to train an N-grams model.
Depending on hyperparameters, 7-12 epochs should be enough.
Please change batch size to suit your compute ability (VRAM).
"""

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# np.set_printoptions(threshold=np.inf)
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pickle

ngram = 10

with open("Models/vocab.pkl", "rb") as f:
  vocab = pickle.load(f)
with open(f"Models/X_train_{ngram}-gram.pkl", "rb") as f:
  X_train = pickle.load(f)
with open(f"Models/y_train_{ngram}-gram.pkl", "rb") as f:
  y_train = pickle.load(f)
with open(f"Models/X_validate_{ngram}-gram.pkl", "rb") as f:
  X_validate = pickle.load(f)
with open(f"Models/y_validate_{ngram}-gram.pkl", "rb") as f:
  y_validate = pickle.load(f)


# evaluation
def predict_top_k_words(model, context_words, k=10, temperature=1.0):
  context_idxs = torch.tensor([[word_to_idx[w] for w in context_words]], device=device_gpu)
  model.eval()
  with torch.no_grad():
    logits = model(context_idxs).squeeze(0) / temperature
    probabilities = torch.softmax(logits, dim=0)

    top_probabilities, top_indices = torch.topk(probabilities, k)

  top_probabilities = top_probabilities.cpu()
  top_indices = top_indices.cpu()
  return [(idx_to_word[i.item()]) for i in top_indices]

def evaluate(model, X, y):
  correct_count = 0
  for i in range(10000):
    # if (y_test[i] in connection_word_list): continue
    if (y[i] in predict_top_k_words(model, X[i], 3)):
      correct_count += 1
  accuracy = correct_count / 10000
  return(accuracy*100)

# creating tensors
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_count = len(vocab)
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NGramModel(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
    super().__init__()
    self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

    self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=0.3, batch_first=True)

    self.fc = nn.Linear(hidden_dim, vocab_size) # required to map the LSTM's hidden state to the desired representations/dimensionality

  def forward(self, x):
    embeds = self.embeddings(x)
    lstm_out, _ = self.lstm(embeds)
    last_hidden = lstm_out[:, -1, :]  # to ignore n previous words in the output
    out = self.fc(last_hidden)
    return out


X = torch.tensor([[word_to_idx[w] for w in context] for context in X_train])
y = torch.tensor([word_to_idx[content] for content in y_train])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16384, shuffle=True, pin_memory=True)

def train(model, embed_dim, hidden_dim, num_layers):
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
  prev_acc = 0
  for epoch in range(25):
    model.train()
    batch_number = 0
    for batch_X, batch_y in loader:
      batch_number += 1
      batch_X = batch_X.to(device_gpu, non_blocking=True)
      batch_y = batch_y.to(device_gpu, non_blocking=True)

      optimizer.zero_grad()

      logits = model(batch_X)
      loss = loss_func(logits, batch_y)

      loss.backward()
      optimizer.step()

    acc = evaluate(model, X_validate, y_validate)
    print(f"{embed_dim}, {hidden_dim}, {num_layers}: Epoch {epoch} accuracy_3w_validate={acc:.2f} accuracy_3w_train={evaluate(model, X_train, y_train):.2f}")
    with open(f"Models/trained_model_{ngram}-gram_embed_dim-{embed_dim}_hidden_dim-{hidden_dim}_num_layers-{num_layers}_epoch-{epoch+1}.pkl", "wb") as f:
      pickle.dump(model, f)
    # if (acc <= prev_acc): break
    # prev_acc = acc

  



# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=150, hidden_dim=96, num_layers=1).to(device_gpu),
#   150,
#   96,
#   1
# )
# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=200, hidden_dim=96, num_layers=1).to(device_gpu),
#   200,
#   96,
#   1
# )
# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=300, hidden_dim=96, num_layers=1).to(device_gpu),
#   300,
#   96,
#   1
# )
# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=150, hidden_dim=128, num_layers=1).to(device_gpu),
#   150,
#   128,
#   1
# )
# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=200, hidden_dim=128, num_layers=1).to(device_gpu),
#   200,
#   128,
#   1
# )
train(
  NGramModel(vocab_size=vocab_count, embed_dim=300, hidden_dim=128, num_layers=1).to(device_gpu),
  300,
  128,
  1
)
# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=150, hidden_dim=96, num_layers=1).to(device_gpu),
#   150,
#   96,
#   2
# )
# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=200, hidden_dim=96, num_layers=2).to(device_gpu),
#   200,
#   96,
#   2
# )
# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=300, hidden_dim=96, num_layers=2).to(device_gpu),
#   300,
#   96,
#   2
# )
# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=150, hidden_dim=128, num_layers=2).to(device_gpu),
#   150,
#   128,
#   2
# )
# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=200, hidden_dim=128, num_layers=2).to(device_gpu),
#   200,
#   128,
#   2
# )
# train(
#   NGramModel(vocab_size=vocab_count, embed_dim=300, hidden_dim=128, num_layers=2).to(device_gpu),
#   300,
#   128,
#   2
# )