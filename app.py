# Web app to deploy model

from flask import Flask, render_template, render_template_string, request
import torch

app = Flask(__name__, template_folder='webapp/templates', static_folder='webapp/static')

from webapp.scripts import *

def load_model():
    """Loads model for text generation.

    Returns:
        Model and vocabulary objects.
    
    """
    # Load vocab
    vocab = torch.load('webapp/static/model/vocab')

    # Model parameters
    BATCH_SIZE = 512
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 512
    OUTPUT_DIM = len(vocab)
    N_LAYERS = 2
    BIDIRECTIONAL = False
    DROPOUT = 0.5
    PAD_IDX = vocab.stoi['<pad>']

    # Load model
    model = TextGenerator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
    model.load_state_dict(torch.load('webapp/static/model/model_params.pt', map_location=device))

    return model, vocab.stoi, vocab.itos


# Load model
model, vocab_stoi, vocab_itos = load_model()

@app.route('/', methods=['GET', 'POST'])
def generate():

    thumb = request.base_url + 'preview/static/img/thumbnail.png'

    if request.method == 'POST':
        seed_text = request.form['seed_text']
        gen_length = int(request.form['gen_length'])

        generated = generate_text(model, gen_length, seed_text, vocab_stoi, vocab_itos, repeats=False)
        return render_template('generated.html', generated_text=generated, thumb=thumb)

    elif request.method == 'GET':
        return render_template('main.html', thumb=thumb)

@app.route('/preview')
def preview():
    return render_template('preview.html')
    #return app.send_static_file('img/thumbnail.png')

if __name__ == '__main__':
    main.run(debug=True)