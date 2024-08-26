import argparse
import torch
import tritonclient.grpc as grpcclient

from src.corpus_loader import fetch_and_load_corpus
from src.utils import embedding_from_prompt

client = grpcclient.InferenceServerClient(url="localhost:8001")

def inference(url, prompt_text, sequence_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = fetch_and_load_corpus(url)

    vocab = sorted(set(corpus))
    vocab_size = len(vocab)

    char_to_index = {char: idx for idx, char in enumerate(vocab)}
    index_to_char = {idx: char for idx, char in enumerate(vocab)}

    prompt_embedding = embedding_from_prompt(prompt_text[-sequence_size:].lower(), char_to_index, vocab_size)
    prompt_embedding = prompt_embedding.unsqueeze(0).to(device)

    inputs = [grpcclient.InferInput('input', prompt_embedding.shape, "FP32")]
    inputs[0].set_data_from_numpy(prompt_embedding.numpy())

    outputs = [grpcclient.InferRequestedOutput('output')]

    response = client.infer("gru", inputs, outputs=outputs)
    response_output = response.as_numpy('output')
    
    logits = torch.from_numpy(response_output)
    logits_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()
    char_idx = torch.multinomial(logits_probs, 1).item()
    char = index_to_char[char_idx]

    print(f"Next character after '{prompt_text}' is '{char}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference of RNN-based model trained on the specified text corpus.")
    
    parser.add_argument('--url', type=str, default='https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
                        help='URL to fetch the text corpus')
    parser.add_argument('--prompt_text', type=str, default='Hello my dear darling and princess, I am fond of you, and you know it very ',
                        help='Text to use as a basis for text generation (e.g., "Forecasting for you ")')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')

    args = parser.parse_args()

    inference(args.url, args.prompt_text, args.sequence_size)