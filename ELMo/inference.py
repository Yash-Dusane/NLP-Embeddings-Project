import torch
import torch.nn as nn
import pickle
import sys
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# ELMo definition (unchanged)
# --------------------------
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

# --------------------------
# CBOW and Skipgram definitions
# --------------------------
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        return self.embedding(x)

class Skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100):
        super(Skipgram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        return self.embedding(x)

# --------------------------
# SVD embedding wrapper
# --------------------------
class SVDEmbedding(nn.Module):
    def __init__(self, embeddings):
        super(SVDEmbedding, self).__init__()
        self.register_buffer('embeddings', torch.tensor(embeddings, dtype=torch.float32))
    
    def forward(self, x):
        return self.embeddings[x]

# --------------------------
# Dataset for inference
# --------------------------
class AGNewsDataset(Dataset):
    def __init__(self, word_to_id, max_len=30):
        self.word_to_id = word_to_id
        self.max_len = max_len
    
    def encode_text(self, text):
        text = text.lower().split()
        encoded = [self.word_to_id.get(word, 0) for word in text][:self.max_len]
        return torch.tensor(encoded, dtype=torch.long)
    
    def __getitem__(self, idx):
        pass  # Not used in inference

# --------------------------
# Classifier definition with configurable emb_dim
# --------------------------
class NewsClassifier(nn.Module):
    def __init__(self, emb_dim=128, hidden_dim=256, num_classes=4, lambda_mode='trainable'):
        super(NewsClassifier, self).__init__()
        self.lambda_mode = lambda_mode
        if lambda_mode == 'trainable':
            self.lambdas = nn.Parameter(torch.randn(3))
        elif lambda_mode == 'frozen':
            self.lambdas = torch.randn(3, requires_grad=False)
        elif lambda_mode == 'learnable_function':
            self.lambda_function = nn.Sequential(
                nn.Linear(3 * hidden_dim * 2, 768),
                nn.ReLU(),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, hidden_dim * 2)
            )
        
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        # Projection layer uses emb_dim from embeddings
        self.projection = nn.Linear(emb_dim, hidden_dim * 2)
        
    def forward(self, x, embeddings):
        # Use embeddings module to get word representations.
        out = embeddings(x)
        if isinstance(out, tuple):  # ELMo returns (lstm_out1, lstm_out2)
            lstm_out1, lstm_out2 = out
            emb = embeddings.embedding(x)
            x_proj = self.projection(emb)
        else:
            # For CBOW, Skipgram, or SVD, assume output is (batch, seq, emb_dim)
            x_proj = self.projection(out)
            lstm_out1 = x_proj
            lstm_out2 = x_proj

        if self.lambda_mode == 'learnable_function':
            combined = self.lambda_function(torch.cat((lstm_out1, lstm_out2, x_proj), dim=-1))
            e_final = combined
        else:
            e_final = self.lambdas[0] * x_proj + self.lambdas[1] * lstm_out1 + self.lambdas[2] * lstm_out2

        lstm_out, _ = self.lstm(e_final)
        logits = self.fc(lstm_out[:, -1, :])
        return logits
# --------------------------
# Helper function to adapt state dict shapes
# --------------------------
def adapt_state_dict(current_state, loaded_state):
    new_state = {}
    for key, cur_val in current_state.items():
        if key in loaded_state:
            loaded_val = loaded_state[key]
            if cur_val.shape == loaded_val.shape:
                new_state[key] = loaded_val
            else:
                # Create new tensor with target shape and copy overlapping elements
                target_shape = cur_val.shape
                new_tensor = torch.zeros(target_shape, dtype=loaded_val.dtype, device=loaded_val.device)
                min_shape = tuple(min(a, b) for a, b in zip(target_shape, loaded_val.shape))
                slices = tuple(slice(0, s) for s in min_shape)
                new_tensor[slices] = loaded_val[slices]
                new_state[key] = new_tensor
        else:
            new_state[key] = cur_val
    return new_state

# --------------------------
# Inference function
# --------------------------
def inference(model, description, word_to_id, embeddings):
    dataset = AGNewsDataset(word_to_id)
    encoded_input = dataset.encode_text(description).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(encoded_input, embeddings)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    return probabilities

# --------------------------
# Main function
# --------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python inference.py <saved classifier model path> <description>")
        sys.exit(1)
    
    classifier_model_path = sys.argv[1]
    description = sys.argv[2]

    # Load word_to_id dictionary from file
    with open("word_to_id.pkl", "rb") as f:
        word_to_id = pickle.load(f)
    
    # Determine embedding type based on classifier model filename.
    if 'cbow' in classifier_model_path.lower():
        embedding_dict = torch.load("cbow.pt", map_location=device, weights_only=False)
        if len(embedding_dict["word_to_id"]) != len(word_to_id):
            print("Vocabulary size mismatch: updating word_to_id from CBOW checkpoint.")
            word_to_id = embedding_dict["word_to_id"]
        embedding_weights = torch.tensor(embedding_dict["word_vectors"], dtype=torch.float32, device=device)
        emb_dim = embedding_weights.size(1)
        embedding_model = CBOW(len(word_to_id), embedding_dim=emb_dim).to(device)
        with torch.no_grad():
            embedding_model.embedding.weight.copy_(embedding_weights)
        embedding_model.eval()
        for param in embedding_model.parameters():
            param.requires_grad = False
    elif 'skipgram' in classifier_model_path.lower():
        embedding_dict = torch.load("skipgram.pt", map_location=device, weights_only=False)
        if len(embedding_dict["word_to_id"]) != len(word_to_id):
            print("Vocabulary size mismatch: updating word_to_id from Skipgram checkpoint.")
            word_to_id = embedding_dict["word_to_id"]
        embedding_weights = torch.tensor(embedding_dict["word_vectors"], dtype=torch.float32, device=device)
        emb_dim = embedding_weights.size(1)
        embedding_model = Skipgram(len(word_to_id), embedding_dim=emb_dim).to(device)
        with torch.no_grad():
            embedding_model.embedding.weight.copy_(embedding_weights)
        embedding_model.eval()
        for param in embedding_model.parameters():
            param.requires_grad = False
    elif 'svd' in classifier_model_path.lower():
        embedding_dict = torch.load("svd.pt", map_location=device, weights_only=False)
        if len(embedding_dict["word_to_id"]) != len(word_to_id):
            print("Vocabulary size mismatch: updating word_to_id from SVD checkpoint.")
            word_to_id = embedding_dict["word_to_id"]
        embedding_weights = torch.tensor(embedding_dict["word_vectors"], dtype=torch.float32, device=device)
        emb_dim = embedding_weights.size(1)
        embedding_model = SVDEmbedding(embedding_weights).to(device)
        embedding_model.eval()
    else:
        # Default to ELMo
        emb_dim = 128
        embedding_model = ELMo(len(word_to_id), embedding_dim=emb_dim).to(device)
        embedding_model.load_state_dict(torch.load("bilstm.pt", map_location=device))
        embedding_model.eval()
        for param in embedding_model.parameters():
            param.requires_grad = False

    # Create classifier using the determined emb_dim.
    classifier = NewsClassifier(emb_dim=emb_dim).to(device)
    loaded_state = torch.load(classifier_model_path, map_location=device)
    adapted_state = adapt_state_dict(classifier.state_dict(), loaded_state)
    classifier.load_state_dict(adapted_state, strict=False)
    
    # Run inference
    probabilities = inference(classifier, description, word_to_id, embedding_model)
    
    # Print class probabilities
    for i, prob in enumerate(probabilities[0]):
        print(f"class-{i+1} {prob:.1f}")

if __name__ == '__main__':
    main()
