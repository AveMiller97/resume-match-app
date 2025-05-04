import os
import nltk

# Step 1: Set a consistent nltk_data directory (within your project folder)
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Step 2: Download necessary NLTK resources if not already available
def download_nltk_resources():
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, download_dir=NLTK_DATA_DIR)

# Run this before using nltk resources
download_nltk_resources()

# Step 3: Now safely import and use NLTK components
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Example usage
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words('english'))
    return ' '.join([t for t in tokens if t not in stop_words])
