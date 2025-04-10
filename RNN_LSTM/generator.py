import torch
import torch.nn as nn
import re
from collections import Counter
import sys

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences
# Load vocabulary
def build_vocab(sentences):
    words = [word for sentence in sentences for word in sentence.split()]
    word_counts = Counter(words)
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), start=2)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab
def load_vocab(corpus_path):
    file_path = corpus_path
    sentences = load_corpus(file_path)
    train_sentences = [" ".join(re.sub(r"[^\w]", " ", sentence).split()) for sentence in sentences]
    vocab = build_vocab(train_sentences)
    return vocab

# Load model function
def load_model(model_class, model_path, vocab_size, embed_size=128, hidden_size=256, n_gram=5):
    model = model_class(vocab_size, embed_size, hidden_size) if model_class != FFNNLanguageModel else model_class(vocab_size, embed_size, hidden_size, n_gram)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict next word
def predict_next_word(model, vocab, input_text, top_k=5, n_gram=5):
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    input_sequence = [vocab.get(word, vocab["<UNK>"]) for word in input_text.split()]
    
    # Only adjust input length for FFNN
    if isinstance(model, FFNNLanguageModel):  
        if len(input_sequence) < n_gram:
            input_sequence = [vocab["<PAD>"]] * (n_gram - len(input_sequence)) + input_sequence  # Pad left
        elif len(input_sequence) > n_gram:
            input_sequence = input_sequence[-n_gram:]  # Truncate left

    input_tensor = torch.tensor(input_sequence, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=-1).squeeze()
        top_indices = torch.topk(probabilities, top_k).indices.tolist()
    
    predictions = [(reverse_vocab[idx], probabilities[idx].item()) for idx in top_indices]
    
    # Print words and probabilities
    for word, prob in predictions:
        print(f"{word}: {prob:.4f}")
    
    return predictions
class FFNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_gram):
        super(FFNNLanguageModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.fc1 = nn.Linear(n_gram * embed_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x).view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# RNN Model
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNLanguageModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# LSTM Model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTMLanguageModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
# Main Execution
def main():
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])
    vocab = load_vocab(corpus_path)
    vocab_size = len(vocab)
    print(vocab_size)
    # Load models
    x=""
    if corpus_path=='Pride and Prejudice - Jane Austen.txt':
        x="PP"
    if corpus_path=='Ulysses - James Joyce.txt':
        x="Uly"
    
    ffnn_model = load_model(FFNNLanguageModel, f"{x}_ffnn_model.pt", vocab_size)
    rnn_model = load_model(RNNLanguageModel, f"{x}_rnn_model.pt", vocab_size)
    lstm_model = load_model(LSTMLanguageModel, f"{x}_lstm_model.pt", vocab_size)
    # Test prediction
    while True:
        test_sentence=input("Input sentence: ").strip().lower()
        if lm_type=='-f':
            print("FFNN Predictions:", predict_next_word(ffnn_model, vocab, test_sentence,k))
        if lm_type=='-r':
            print("RNN Predictions:", predict_next_word(rnn_model, vocab, test_sentence,k))
        if lm_type=='-l':
            print("LSTM Predictions:", predict_next_word(lstm_model, vocab, test_sentence,k))

if __name__ == "__main__":
    main()
