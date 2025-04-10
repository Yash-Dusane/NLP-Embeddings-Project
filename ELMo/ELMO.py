import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from nltk.corpus import brown
import nltk
import random
import pickle
from tqdm import tqdm

nltk.download('brown')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sentences = brown.sents()
processed_sentences = [[word.lower() for word in sentence] for sentence in sentences]
word_counts = Counter(word for sentence in processed_sentences for word in sentence)
vocab = {word for word, count in word_counts.items() if count >= 5}
word_to_id = {word: i for i, word in enumerate(vocab, start=1)} 
word_to_id['<pad>'] = 0
id_to_word = {i: word for word, i in word_to_id.items()}
vocab_size = len(word_to_id)

with open("word_to_id.pkl", "wb") as f:
    pickle.dump(word_to_id, f)
# print(len(word_to_id))


class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(ELMo, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.output_layer = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x):
        emb = self.embedding(x)
        lstm_out1, _ = self.lstm1(emb)
        lstm_out2, _ = self.lstm2(lstm_out1)
        weighted_out = self.gamma * lstm_out2 + self.beta * lstm_out1
        logits = self.output_layer(weighted_out)
        return logits

model = ELMo(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)


def prepare_data(sentences, word_to_id, max_len=30):
    data = []
    for sentence in sentences:
        encoded = [word_to_id[word] for word in sentence if word in word_to_id]
        if len(encoded) < 2:
            continue  
        encoded = encoded[:max_len]
        input_seq_forward = torch.tensor(encoded[:-1], dtype=torch.long)
        target_seq_forward = torch.tensor(encoded[1:], dtype=torch.long)
        input_seq_backward = torch.tensor(encoded[1:], dtype=torch.long)
        target_seq_backward = torch.tensor(encoded[:-1], dtype=torch.long)
        data.append((input_seq_forward, target_seq_forward, input_seq_backward, target_seq_backward))
    return data

data = prepare_data(processed_sentences, word_to_id)
random.shuffle(data)

def train(model, data, epochs=5, batch_size=32):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            inputs_fwd, targets_fwd, inputs_bwd, targets_bwd = zip(*batch)
            
            inputs_fwd = nn.utils.rnn.pad_sequence(inputs_fwd, batch_first=True, padding_value=0).to(device)
            targets_fwd = nn.utils.rnn.pad_sequence(targets_fwd, batch_first=True, padding_value=0).to(device)
            inputs_bwd = nn.utils.rnn.pad_sequence(inputs_bwd, batch_first=True, padding_value=0).to(device)
            targets_bwd = nn.utils.rnn.pad_sequence(targets_bwd, batch_first=True, padding_value=0).to(device)
            
            optimizer.zero_grad()
            logits_fwd = model(inputs_fwd)
            logits_bwd = model(inputs_bwd)
            
            loss_fwd = criterion(logits_fwd.view(-1, vocab_size), targets_fwd.view(-1))
            loss_bwd = criterion(logits_bwd.view(-1, vocab_size), targets_bwd.view(-1))
            
            loss = loss_fwd + loss_bwd
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data):.4f}")

train(model, data, epochs=5)
torch.save(model.state_dict(), "bilstm.pt")
