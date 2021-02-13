# Isolated versions of imports and functions required by webapp

import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from IPython.display import clear_output

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
  
def generate_text(model, gen_len, text, token2int, int2token, end_words=False, repeats=False):
    """Generates text from starting text.

    Args:
        model: model to use for prediction.
        gen_len: length of desired generation.
        text: text to generate from.
        token2int: token to integer dictionary for vocab.
        int2token: integer to token dictionary for vocab.
        end_words: whether or not to restrict end words.
        repeats: whether or not to restrict repeated words.
    Returns:
        Generated text.

    """
    model.eval()

    # Batch size is 1 (single text example)
    hidden = model.init_hidden(1)

    # Split text into tokens
    tokens = text.lower().split()

    # Pass wordlist if checking for repeats
    if repeats:
        wordlist = tokens
    else:
        wordlist = None

    # Predict next word
    for token in tokens:
        token, hidden = predict_next(model, token, token2int, int2token, hidden, wordlist)
          
    tokens.append(token)

    # Predict subsequent
    for i in range(gen_len - 1):
        token, hidden = predict_next(model, tokens[-1], token2int, int2token, hidden, wordlist)
        tokens.append(token)
  
    return ' '.join(tokens)