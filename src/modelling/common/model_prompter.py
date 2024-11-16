from collections.abc import Callable

import torch

from src.constants.corpus import END_OF_THOUGHT_TOKEN, UNKNOWN_TOKEN
from src.constants.device import device
from src.modelling.common.model_loader import load_model_from_archive
from src.modelling.common.model_trainer import CorpusUtils, Metadata


def apply_temperature(logits_probs, temperature: float = 0.0):
    logits_probs = logits_probs / (temperature + 1e-6)
    return torch.softmax(logits_probs, dim=-1)


def apply_repetition_penalty(logits_probs, generated_indices, penalty: float = 1.0):
    for index in generated_indices:
        logits_probs[index] /= penalty
    return logits_probs


def top_k_sampling(logits_probs, k: int = 10):
    top_k_probs, top_k_indices = torch.topk(logits_probs, k)
    top_k_probs = top_k_probs / top_k_probs.sum()

    token_index = torch.multinomial(top_k_probs, 1).item()

    return top_k_indices[token_index].item()


def top_p_sampling(logits_probs, p: float = 0.9):
    sorted_probs, sorted_indices = torch.sort(logits_probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff_index = torch.where(cumulative_probs > p)[0][0].item()

    top_p_probs = sorted_probs[: cutoff_index + 1]
    top_p_probs = top_p_probs / top_p_probs.sum()
    top_p_indices = sorted_indices[: cutoff_index + 1]

    token_index = torch.multinomial(top_p_probs, 1).item()

    return top_p_indices[token_index].item()


def make_prompter(
    metadata: Metadata,
    input_to_padded: Callable,
    corpus_utils: CorpusUtils,
):
    model = load_model_from_archive(metadata)

    def prompt(input_text: str, sequence_size, output_size: int = 255) -> str:
        output_text = input_text.lower().split()
        padded_input = input_to_padded(input_text, sequence_size)

        indices = corpus_utils.input_to_index(padded_input).to(device)

        model.eval()

        token = None
        i = 0
        while i < output_size and token != END_OF_THOUGHT_TOKEN:
            with torch.no_grad():
                logits = model(indices)

                logits_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()
                # pylint: disable=using-constant-test
                if False:
                    logits_probs = apply_temperature(logits_probs, temperature=0.0)
                    logits_probs = apply_repetition_penalty(
                        logits_probs,
                        [
                            corpus_utils.token_to_index.get(char, corpus_utils.token_to_index[UNKNOWN_TOKEN])
                            for char in output_text
                        ],
                        penalty=1.0,
                    )

                token_index = top_k_sampling(logits_probs, k=10)
                token = corpus_utils.index_to_token[token_index]

            if token != UNKNOWN_TOKEN:
                output_text.append(token)

            next_index = corpus_utils.input_to_index(token).to(device)

            indices = torch.cat((indices[:, -sequence_size + 1 :], next_index), dim=1)

            i += 1

        return " ".join(output_text)

    return prompt
