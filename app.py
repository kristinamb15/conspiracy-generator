# Web app to deploy model

from flask import Flask, render_template, request

# Make sure imports work appropriately
import sys
sys.path.insert(0, './src')

# Import modules from src
from app_scripts import *
from neural_network import *

app = Flask(__name__)

def load_model():
    # Load model
    model = torch.load('model/model.pt')

    # Load vocab
    vocab = torch.load('model/vocab')

    return model, vocab.stoi, vocab.itos


@app.route('/', methods=['GET', 'POST'])
def generate():
    # Load model
    model, vocab_stoi, vocab_itos = load_model()

    if request.method == 'POST':
        seed_text = request.form['seed_text']
        gen_length = int(request.form['gen_length'])        

        generated = generate_text(model, gen_length, seed_text, vocab_stoi, vocab_itos, end_words=False, repeats=False)
        return render_template('generated.html', generated_text=generated)

    elif request.method == 'GET':
        return render_template('main.html')

if __name__ == '__main__':
    app.run()