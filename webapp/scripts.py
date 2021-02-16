#Neural network and text generation functions

# Standard imports
import numpy as np
import random
import string

# NN imports
import torch
from torch import nn
import torch.nn.functional as F

# Torch device
device = 'cpu'

# Define NN
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, hidden):
        # text dimensions: [sent len, batch size]

        embedded = self.dropout(self.embedding(text))
        # embedded dimensions: [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded, hidden)

        output = output.reshape(-1, self.hidden_dim * 2 if self.bidirectional else self.hidden_dim)
        # output dimensions = [sent len, batch size, hid dim * num directions]
        # hidden state is really (hidden, cell) where:    
        # hidden dimensions = [num layers * num directions, batch size, hid dim]
        # cell dimensions: = [num layers * num directions, batch size, hid dim]

        return self.fc(output), hidden
        # hidden = [batch size, hid dim * num directions]

    # Initialize hidden state
    def init_hidden(self, batch_size):
        # Two tensors of size [n_layers * 2, batch_size, hidden_dim]
        return (torch.zeros(2 * self.n_layers if self.bidirectional else self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(2* self.n_layers if self.bidirectional else self.n_layers, batch_size, self.hidden_dim).to(device))

def predict_next(model, word, token2int, int2token, hidden=None, wordlist=None):
    """Predicts next word from given word.

    Args:
        model: model to use for prediction.
        word: word to use as generator.
        token2int: token to integer dictionary for vocab.
        int2token: integer to token dictionary for vocab.
        hidden: hidden state (optional).
        wordlist: if predicted word is in word list already, get a different one.
    Returns:
        Predicted next word.
    
    """
    # Create tensor from word
    x = np.array([[token2int[word]]])
    x = torch.from_numpy(x).to(device).long()

    # Detach hidden state
    hidden = tuple([each.data for each in hidden])

    # Get prediction from model
    output, hidden = model(x, hidden)

    # Get probabilities for words in vocab
    probs = F.softmax(output, dim=1).data.cpu()
    probs = probs.numpy()
    probs = probs.flatten()

    # Get indices of top five probabilities
    top_n = probs.argsort()[-5:]

    # Get top 5 words
    preds = [int2token[i] for i in top_n]

    # Randomly select one of the five
    pred = preds[random.sample([0,1,2,3,4], 1)[0]]
    
    # If wordlist is given, check if predicted word is already in word list - if so, get another
    if wordlist:
        # if all predicted words are in the wordlist, return prediction already made
        if set(preds).issubset(set(wordlist)):
            pass
        else:
            while pred in wordlist:
                sample = top_n[random.sample([0,1,2,3,4], 1)[0]]
                pred = int2token[sample]

    return pred, hidden
  
def generate_text(model, gen_len, text, token2int, int2token, repeats=False):
    """Generates text from starting text.

    Args:
        model: model to use for prediction.
        gen_len: length of desired generation.
        text: text to generate from.
        token2int: token to integer dictionary for vocab.
        int2token: integer to token dictionary for vocab.
        repeats: whether or not to restrict repeated words.
    Returns:
        Generated text.

    """
    model.eval()

    # Batch size is 1 (single text example)
    hidden = model.init_hidden(1)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Split text into tokens
    tokens = text.lower().split()

    # Pass wordlist if checking for repeats
    if repeats:
        wordlist = tokens + ['<unk>', '<pad>']
    else:
        wordlist = ['<unk>', '<pad>']

    # Predict next word
    for token in tokens:
        token, hidden = predict_next(model, token, token2int, int2token, hidden, wordlist)
          
    tokens.append(token)

    # Predict subsequent
    for i in range(gen_len - 1):
        token, hidden = predict_next(model, tokens[-1], token2int, int2token, hidden, wordlist)
        tokens.append(token)
  
    return ' '.join(tokens)

if __name__ == '__main__':
    vocab = torch.load('webapp/static/model/vocab')

    BATCH_SIZE = 512
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 512
    OUTPUT_DIM = len(vocab)
    N_LAYERS = 2
    BIDIRECTIONAL = False
    DROPOUT = 0.5
    PAD_IDX = vocab.stoi['<pad>']

    model = TextGenerator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
    model.load_state_dict(torch.load('webapp/static/model/model_params.pt', map_location=device))

    pred, hid = predict_next(model, 'word', vocab.stoi, vocab.itos, hidden=model.init_hidden(1), wordlist=None)
    print(pred)
    gen = generate_text(model, 5, "test test", vocab.stoi, vocab.itos, repeats=False)
    print(gen)