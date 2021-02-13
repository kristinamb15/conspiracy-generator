# Clean text
from utilities import *

def process_text(text, tokens=True, hyphen=True, stops=None, lemm=False):
    """Cleans text.

    Args:
        text: text to process.
        tokens: set to True to return list of tokens.
        hyphen: whether or not to remove hyphens.
        stops: list of extra stop words to remove.
        lemm: whether or not to lemmatize.
    Returns:
        Clean text or tokens.

    """  
    # Lowercase
    text = text.lower()

    # Remove hyphen
    if hyphen:
        text = re.sub(r'[-–—]', ' ', text)

    # Remove sentence punctuation
    text = ''.join([char for char in text if char not in '. ? !'.split()])

    # Remove stop words
    if stops:
        text = ' '.join([word for word in text.split() if word.strip() not in stops])

    # Remove citations of the form '[81]'
    text = re.sub(r'\[\d+\]', '', text)

    # Remove '[citation needed]'
    text = re.sub(r'\[\w*\s*citation needed]', '', text)

    # Remove all punctuation and non-english non-alphanumeric characters that are not whitespace or hyphens
    text = re.sub(r'[^a-z\s\d\.\?\!-]', '', text)

    if lemm:
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Lemmatize
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    # One more pass at stop words
    if stops:
        text = ' '.join([word for word in text.split() if word not in stops])
        
    # Strip extra whitespace from words
    tok = [word.strip() for word in text.split() if len(word) > 1]

    if tokens:
        return tok
    else:
        return ' '.join(tok)

if __name__ == '__main__':
    # Load dataframe
    conspiracy_df = pd.read_csv(path.join(raw_data, 'conspiracy_df.csv'))

    # Create list of stop words to remove
    stops = stopwords.words('english')
    new_stops = ['conspiracy', 'conspiracies', 'theory', 'theories', 'believe']
    stops.extend(new_stops)

    name = str(input('Name: '))
    rem_hyphen = bool(int(input('Remove hyphens (1 for True, 0 for False): ')))
    rem_stops = bool(int(input('Remove stop words (1 for True, 0 for False): ')))

    if rem_stops:
        stop_list = stops
    else:
        stop_list = None    

    # Apply to dataframe and append
    conspiracy_df[f'clean_text_{name}'] = conspiracy_df['text'].apply(lambda x: process_text(x, tokens=False, hyphen=rem_hyphen, stops=stop_list))

    # Save file
    conspiracy_df.to_csv(path.join(raw_data, 'conspiracy_df.csv'), index=False)

    # Concatenate text columns and save to file
    raw_text = ' '.join(conspiracy_df[f'clean_text_{name}'])
    with open(path.join(raw_data, f'raw_text_{name}.txt'), 'w') as txt:
        txt.write(raw_text)