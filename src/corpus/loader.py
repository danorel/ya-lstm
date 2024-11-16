import pathlib
import re

import nltk
from nltk.corpus import words
import requests

from src.constants.metadata import CORPUS_DATA_DIR, CORPUS_NLTK_DICTIONARY_DIR

# Download a list of valid English words from nltk
nltk.data.path.append(CORPUS_NLTK_DICTIONARY_DIR)
nltk.download("words", download_dir=CORPUS_NLTK_DICTIONARY_DIR)

# Load a list of valid English words from dictionary
valid_words = set(word.lower() for word in words.words())


def fetch_text(url) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "text/plain" in content_type:
        return response.text
    raise ValueError(f"Expected text/plain content, but got {content_type}")


def preprocess(corpus: str):
    corpus = re.sub(r"[^a-zA-Z\s]", "", corpus)

    # Tokenize the corpus into words
    tokenized = corpus.split()

    # Conditionally lowercase words if they are valid English words
    processed_tokens = [word.lower() if word.lower() in valid_words else word for word in tokenized]

    # Remove words which are only upper case
    processed_tokens = [word for word in processed_tokens if word.islower()]

    return " ".join(processed_tokens)


def fetch_and_load_corpus(url: str) -> str:
    name = url.split("/")[-1]
    path = pathlib.Path(f"{CORPUS_DATA_DIR}/{name}/corpus.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, "r+", encoding="utf-8") as f:
            corpus = f.read()
    else:
        corpus = fetch_text(url)
        with open(path, "w+", encoding="utf-8") as f:
            f.write(corpus)
    return preprocess(corpus)
