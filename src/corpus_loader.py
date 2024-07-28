import pathlib
import requests

from src.constants import CORPUS_DIR

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
    corpus = corpus.lower()
    return corpus

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