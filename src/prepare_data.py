# Prepare data for neural network training
from utilities import *


def create_sequences(text, start=0, stop=None, length=5):
    """Creates sequences of given length from text.

    Args:
        text: text to split into sequences.
        start: starting word index.
        stop: stopping word index.
        length: desired length of sequences.
    Returns:
        List of sequences.

    """
    sequences = []

    text = text.split()[start:stop]
    text = ' '.join(text)

    # If text isn't longer than length, just return the text
    if len(text) <= length:
        sequences = [text]
    else:
        for i in range(length, len(text.split())):
            seq = text.split()[i - length: i + 1]
            sequences.append(' '.join(seq))
            
    return sequences

def prep_data(sequences, to_df=True):
    """Returns text and targets for list of sequences.

    Args:
        sequences: text sequences to split into text/target.
        to_df: set to True to return dataframe.
    Returns:
        Lists of text and targets or dataframe.

    """
    x = []
    y = []

    # The target for a sequence of text will be the sequence shifted forward one word
    for seq in sequences:
        x.append(' '.join(seq.split()[:-1]))
        y.append(' '.join(seq.split()[1:]))

    if to_df:
        df = pd.DataFrame({'text': x, 'target': y})
        return df
    else:
        return x, y

if __name__ == '__main__':
    # Load dataframe and grab text (alternatively, this can be collected from the corresponding text files)
    conspiracy_df = pd.read_csv(path.join(raw_data, 'conspiracy_df.csv'))

    col = str(input('Column of text (clean_text_?): '))
    raw_text = ' '.join(conspiracy_df[f'clean_text_{name}'])

    # Create text/target sequences of given length for both sets of text and save
    seq_len = int(input('Sequence length: '))

    sequences = create_sequences(raw_text, 0, None, seq_len)
    df = prep_data(sequences, to_df=True)
    df.to_csv(path.join(proc_data, f'conspiracy_text_{name}.csv'), index=False)

    # Concatenate both dataframes and save
    #df = pd.concat([df1, df2], ignore_index=True)
    #df.to_csv(path.join(proc_data, 'conspiracy_text_all.csv'), index=False)