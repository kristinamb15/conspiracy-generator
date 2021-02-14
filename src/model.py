# Define model
from utilities import *
from neural_network import *

from datetime import datetime

# Visualization style
sns.set_style('darkgrid')

SEED = torch.manual_seed(108)

def set_args():
    """Sets certain arguments for the model.

    Returns:
        Model arguments.

     """  
    BATCH_SIZE = int(input('Batch size: '))
    HIDDEN_DIM = int(input('Hidden dimension: '))
    N_LAYERS = int(input('Number of layers: '))
    BIDIRECTIONAL = bool(int(input('Bidirectional (1 for True, 0 for False): ')))
    DROPOUT = float(input('Dropout: '))
    N_EPOCHS = int(input('Number of epochs: '))

    return BATCH_SIZE, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, N_EPOCHS


if __name__ == '__main__':
    # Name model based on timestamp
    model_name = datetime.now().strftime('%m-%d-%Y_(%H.%M.%S)')
    os.mkdir(f'models/{model_name}')
    model_path = f'models/{model_name}'

    print('Model name: ', model_name)
    print('Model path: ', model_path)

    # Get dataframe to train on
    data_df = str(input('Data: '))
    print('\nTrained on: ', data_df)

    # Create Field objects
    TEXT = data.Field(batch_first=True)
    TARGET = data.Field(batch_first=True)

    # Create tuples representing the columns
    fields = [('text', TEXT), ('target', TARGET)]

    # Load the dataset
    text_data = data.TabularDataset(
    path = path.join(proc_data, f'{data_df}.csv'),
    format = 'csv',
    fields = fields,
    skip_header = True
    )

    # Get parameters
    BATCH_SIZE, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, N_EPOCHS = set_args()

    # Split into training/validation
    train_data, valid_data = text_data.split(split_ratio=0.7, random_state = random.seed(SEED))

    # Build vocabulary and save
    TEXT.build_vocab(train_data, vectors = "glove.6B.300d", unk_init = torch.Tensor.normal_)
    TARGET.vocab = TEXT.vocab
    torch.save(TEXT.vocab, f'{model_path}/vocab')

    # Create batch iterators
    train_iterator, valid_iterator = data.Iterator.splits(
        (train_data, valid_data), 
        batch_size=BATCH_SIZE,
        device=device,
        sort=False,
        shuffle=True)

    # Define model
    VOCAB_SIZE = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    OUTPUT_DIM = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = TextGenerator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

    # Prepare pretrained embeddings
    pretrained_embeddings = TEXT.vocab.vectors

    # Replace initial weights of embedding layer with pre-trained embeddings
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # Our <unk> and <pad> tokens have been randomly initialized, but we want them to be zeroed out
    # The <unk> token will be learned, but the <pad> token will not
    UNK_IDX  = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    # Print example data
    print('\nExample data: ', vars(text_data.examples[random.randint(0,1000)]))

    # Print details
    print('\nSize of text vocabulary: ', len(TEXT.vocab))
    print('Size of target vocabulary: ', len(TARGET.vocab))
    print('\n')
    
    details = {
                'BATCH_SIZE': BATCH_SIZE,
                'VOCAB_SIZE': VOCAB_SIZE,
                'EMBEDDING_DIM': EMBEDDING_DIM,
                'HIDDEN_DIM': HIDDEN_DIM,
                'OUTPUT_DIM': OUTPUT_DIM,
                'N_LAYERS': N_LAYERS,
                'BIDIRECTIONAL': BIDIRECTIONAL,
                'DROPOUT': DROPOUT,
                'N_EPOCHS': N_EPOCHS
            }   
    for item in details.items():
        print(item)
    print('\n')

    # Count trainable parameters
    count_parameters(model)
    print('\n')

    # Train model
    t_loss, v_loss = train_model(N_EPOCHS, model, optimizer, criterion, train_iterator, valid_iterator, BATCH_SIZE, training_step, evaluation_step, model_path)

    # Plot loss and save to file
    sns.lineplot(x=[i for i in range(1, N_EPOCHS + 1)], y=t_loss, #label='training loss')
    sns.lineplot(x=[i for i in range(1, N_EPOCHS + 1)], y=v_loss, #label='valid. loss')
    plt.legend()
    plt.savefig(f'{model_path}/loss.png')
    plt.show()

