# Text generating neural network and associated fucntions

# Commented for webapp purposes - uncomment to run as script
# from utilities import *

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

def count_parameters(model):
    """Counts trainable parameters of model.

    Args:
        model: model to use.
    Returns:
        Numer of trainable parameters.

     """       
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {params:,} trainable parameters')

def epoch_time(start_time, end_time):
    """Counts length of time it takes per epoch.

    Args:
        start_time: start of epoch.
        end_time: end of epoch.
    Returns:
        Prints elapsed mins, secs in epoch.

    """ 
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

def training_step(model, iterator, optimizer, criterion, batch_size):
    """Performs a training step, iterating over all examples, one batch at a time.

    Args:
        model: model to use.
        iterator: data iterator.
        optimizer: optimizer to use.
        criterion: loss function to use.
        batch_size: batch size.
    Returns:
        Average training loss for the epoch.

    """ 
    epoch_loss = 0

    # Reset states
    hidden = model.init_hidden(batch_size)

    # Put model in training mode
    model.train()

    for batch in iterator:

        # Don't use incomplete batches
        if batch.batch_size == batch_size:

            # Detach hidden states
            hidden = tuple([each.data for each in hidden])
            
            # Zero gradients
            optimizer.zero_grad()

            # Make predictions
            predictions, hidden = model(batch.text.to(device), hidden)

            # Make predictions
            #predictions = model(batch.text)

            # Calculate loss
            #print('predictions: ', predictions.shape)
            #print('target: ', batch.target.shape)
            loss = criterion(predictions, batch.target.view(-1))

            # Compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            # Update loss for averages
            epoch_loss += loss.item()
  
    return epoch_loss / len(iterator)

def evaluation_step(model, iterator, criterion, batch_size):
    """Performs a validation step.

    Args:
        model: model to use.
        iterator: validation batch iterator.
        criterion: loss function to use.
    Returns:
        Average validation loss for the epoch.

    """ 
    epoch_loss = 0

    # Reset states
    hidden = model.init_hidden(batch_size)

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        for batch in iterator:

            # Don't use incomplete batches
            if batch.batch_size == batch_size:

                # Make predictions
                predictions, hidden = model(batch.text.to(device), hidden)

                # Calculate loss
                loss = criterion(predictions, batch.target.view(-1))

                # Update loss and accuracy for averages
                epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def train_model(n_epochs, model, optimizer, criterion, train_iterator, valid_iterator, batch_size, training_fn, evaluation_fn, model_path):
    """Trains model.

    Args:
        n_epochs: number of epochs.
        model: model to use.
        optimizer: optimizer to use.
        criterion: loss function to use.
        train_iterator: training batch iterator.
        valid_iterator: validation batch iterator.
        batch_size: batch size.
        training_fn: function that defines a single training step.
        evaluation_fn: function that defines a single evaluation step.
        model_path: path to save model/optimizer parameters to.
    Returns:
        Lists of training/validation loss for each epoch.

    """ 
    best_valid_loss = float('inf')
    t_loss = []
    v_loss = []

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = training_fn(model, train_iterator, optimizer, criterion, batch_size)
        valid_loss = evaluation_fn(model, valid_iterator, criterion, batch_size)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # If the val loss is the best, save model
        if valid_loss < best_valid_loss:
            best_valid_loss = best_valid_loss
            torch.save(model.state_dict(), f'{model_path}/model_params.pt')
            torch.save(optimizer.state_dict(), f'{model_path}/optim_params.pt')
            torch.save(model, f'{model_path}/model.pt')
        
        t_loss.append(train_loss)
        v_loss.append(valid_loss)
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    return t_loss, v_loss

if __name__ == '__main__':
    print('CUDA available: ', torch.cuda.is_available())