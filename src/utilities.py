# Functions/variables that get used in other files

# Standard imports
import pandas as pd
import numpy as np
import os

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Scraping Imports
import requests
import bs4
import re

# NLP Imports
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Uncomment if not installed
#nltk.download('stopwords')
#nltk.download('wordnet')

# NN Imports
import time
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data

# Use GPU when available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Basic setup
torch.backends.cudnn.deterministic = True

# Paths
from os import path
raw_data = path.abspath('data/raw')
proc_data = path.abspath('data/processed')
