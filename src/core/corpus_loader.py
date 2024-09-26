import nltk
import pathlib
import re
import requests

from nltk.corpus import words

from src.core.constants import CORPUS_DIR

# Download a list of valid English words from nltk
nltk.data.path.append('./dictionary')
nltk.download('words', download_dir='./dictionary/')

# Load a list of valid English words from dictionary
valid_words = set([word.lower() for word in words.words()])


def fetch_text(url) -> str:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    content_type = response.headers.get('Content-Type', '')
    if 'text/plain' in content_type:
        return response.text
    else:
        raise ValueError(f"Expected text/plain content, but got {content_type}")

def preprocess(corpus: str):
    corpus = re.sub(r'[^a-zA-Z\s]', '', corpus)

    # Tokenize the corpus into words
    tokenized = corpus.split()
    
    # Conditionally lowercase words if they are valid English words
    processed_tokens = [
        word.lower() if word.lower() in valid_words else word
        for word in tokenized
    ]

    # Remove words which are only upper case
    processed_tokens = [
        word
        for word in processed_tokens
        if word.islower()
    ]

    return ' '.join(processed_tokens)

def fetch_and_load_corpus(url: str) -> str:
    filename = url.split('/')[-1]
    filepath = pathlib.Path(f'{CORPUS_DIR}/{filename}')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        with open(filepath, 'r+') as f:
            corpus = f.read()
    else:
        corpus = fetch_text(url)
        with open(filepath, 'w+') as f:
            f.write(corpus)
    return preprocess(corpus)