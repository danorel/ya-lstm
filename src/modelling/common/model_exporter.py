import torch

from src.constants.device import device
from src.modelling.common.model_arguments import CorpusUtils, Metadata
from src.modelling.common.model_loader import load_model_from_artifacts, load_model_to_repository


def make_exporter(metadata: Metadata, corpus_utils: CorpusUtils):
    model = load_model_from_artifacts(metadata)

    def export(sequence_size: int, version: str = "1"):
        load_model_to_repository(
            model,
            metadata,
            version,
            example=torch.randn(1, sequence_size, corpus_utils.vocab_size).to(device),
        )

    return export
