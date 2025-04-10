import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
import pickle
from torch.nn.utils.rnn import pad_sequence

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained ELMo model
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
        return lstm_out1, lstm_out2

# Load vocabulary and pretrained model
with open("word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)
vocab_size = len(word_to_id)

elmo = ELMo(vocab_size).to(device)
elmo.load_state_dict(torch.load("bilstm.pt", map_location=device))
elmo.eval()
for param in elmo.parameters():
    param.requires_grad = False

# Define dataset
class AGNewsDataset(Dataset):
    def __init__(self, csv_path, word_to_id, max_len=30):
        self.data = pd.read_csv(csv_path)
        self.word_to_id = word_to_id
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Description'].lower().split()
        label = self.data.iloc[idx]['Class Index'] - 1  # Ensure zero-based indexing for CrossEntropyLoss
        encoded = [self.word_to_id.get(word, 0) for word in text][:self.max_len]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return inputs_padded, targets_tensor

train_dataset = AGNewsDataset("train.csv", word_to_id)
test_dataset = AGNewsDataset("test.csv", word_to_id)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Classification model
class NewsClassifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=4, lambda_mode='trainable'):
        super(NewsClassifier, self).__init__()
        self.lambda_mode = lambda_mode
        if lambda_mode == 'trainable':
            self.lambdas = nn.Parameter(torch.randn(3))
        elif lambda_mode == 'frozen':
            self.lambdas = torch.randn(3, requires_grad=False)
        elif lambda_mode == 'learnable_function':
           self.lambda_function = nn.Sequential(
            nn.Linear(1536, 768), 
            nn.ReLU(),
            nn.Linear(768, 512), 
            nn.ReLU(),
            nn.Linear(512, hidden_dim * 2)  
        )
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.projection = nn.Linear(128, hidden_dim * 2)
    
    def forward(self, x, elmo):
        lstm_out1, lstm_out2 = elmo(x)
        emb = elmo.embedding(x)
        x_proj = self.projection(emb)
        if self.lambda_mode == 'learnable_function':
            combined = self.lambda_function(torch.cat((lstm_out1, lstm_out2, x_proj), dim=-1))
            e_final = combined
        else:
            e_final = self.lambdas[0] * x_proj + self.lambdas[1] * lstm_out1 + self.lambdas[2] * lstm_out2
        lstm_out, _ = self.lstm(e_final)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

def train_classifier(model, train_loader, test_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs, elmo)
            loss = criterion(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf detected in loss. Skipping batch.")
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    best_accuracy = evaluate_and_save_best(model, test_loader, best_accuracy)
    return best_accuracy

def evaluate_and_save_best(model, test_loader, best_accuracy):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs, elmo)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Final Test Accuracy: {accuracy:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
   
    return best_accuracy

# Train for all lambda modes
global_best_accuracy = 0
for lambda_mode in ['trainable', 'frozen', 'learnable_function']:
    print(f"Training with {lambda_mode} λs:")
    classifier = NewsClassifier(lambda_mode=lambda_mode).to(device)
    best_accuracy = train_classifier(classifier, train_loader, test_loader, epochs=10)
    if best_accuracy > global_best_accuracy:
        global_best_accuracy = best_accuracy
        torch.save(classifier.state_dict(), "classifier.pt")
        print(f"Updated best model saved with accuracy: {global_best_accuracy:.4f}")
