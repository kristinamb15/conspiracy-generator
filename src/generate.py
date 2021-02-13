# Generate text
from utilities import *
from neural_network import TextGenerator

from IPython.display import clear_output

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

def make_conspiracy(text=None, length=None, repeats=False):
    """Asks for seed text and length and generates conspiracy theory.

    Args:
        text: text to generate from.
        length: number of words to add.
        repeats: whether or not to check for repeated words.
    Returns:
        Generated text.

    """
    while True:
        # Clear output
        #os.system('cls')
        clear_output()

        # Gather arguments
        if text is None:
            text = str(input('Enter seed text (lowercase): '))
            length = int(input('Enter number of words to generate: '))
            repeats = bool(int(input('Restrict repeated words (1 for True, 0 for False): ')))

        # Generate text
        generated = generate_text(model, length, text, token2int=vocab.stoi, int2token=vocab.itos, repeats=repeats)
        print('\n' + generated)

        # Ask to generate again, get new input, or quit
        instruct = ''
        while instruct not in [0, 1, 2]:
            instruct = int(input("\nType 0 to generate from the same input.\nType 1 to give new input.\nType 2 to quit.\n"))
            if instruct not in [0, 1, 2]:
                print('\nSorry, that is not a valid choice.')
        if instruct == 2:
            break
        elif instruct == 0:
            # Generate again from same input 
            make_conspiracy(text, length, repeats)
            break
        elif instruct == 1:
            # Ask for new input
            make_conspiracy()
            break

if __name__ == '__main__':
    #os.system('cls')
    clear_output()

    model_path = str(input('Model path: '))

    print('Preparing model...')

    # Load vocab
    vocab = torch.load(f'{model_path}/vocab')

    # Load model
    model = torch.load(f'{model_path}/model.pt')

    # Generate
    make_conspiracy()