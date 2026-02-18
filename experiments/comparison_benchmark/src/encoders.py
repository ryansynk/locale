import math
from itertools import islice
from pathlib import Path

import torch
from sourmash import MinHash, SourmashSignature
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig
from transformers.utils import logging as transformers_logging

from rawbert.modeling.bert_layers import BertModel as DNABertModel
from rawbert.modeling.model import RawBERT
from rawbert.utils.patch import patch_with_flash_lib

from .config import DenseConfig, SourMashConfig


def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


class BaseEncoder:
    def encode(self, sequences: list[str]):
        """Returns a list of feature vectors (or MinHash objects)."""
        raise NotImplementedError


class DenseEncoder(BaseEncoder):
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: DenseConfig, device: str = "cuda"):
        transformers_logging.set_verbosity_error()
        # Load your PyTorch model or DNABERT here
        bert_config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        if not hasattr(bert_config, "pad_token_id") or bert_config.pad_token_id is None:
            bert_config.pad_token_id = 3  # DNABERT Tokenizer [PAD] token id

        if cfg.name == "rawbert":
            assert cfg.checkpoint_path, "No checkpoint provided!"
            checkpoint_path = Path(cfg.checkpoint_path).resolve()
            assert checkpoint_path.is_file(), "Checkpoint does not exist!"
            checkpoint = torch.load(checkpoint_path)
            model = RawBERT(
                pooling="max",
                dim=checkpoint["model_args"]["dim"],
                K=checkpoint["model_args"]["K"],
                m=checkpoint["model_args"]["m"],
                T=checkpoint["model_args"]["T"],
            )
            model.load_state_dict(checkpoint["model"])
            model = model.eval().to(device)
            forward = model.bert_q
        elif cfg.name == "dnabert":
            model = DNABertModel.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                trust_remote_code=True,
                config=bert_config,
            )
            if hasattr(model, "pooler") and model.pooler is not None:
                del model.pooler
                model.pooler = None
            patch_with_flash_lib(model)
            model = model.eval().to(device)
            forward = model
        else:
            raise ValueError(
                f"Expected model_name to be 'rawbert' or 'dnabert', got: {cfg.name}"
            )

        self.model = model
        self.batch_size = cfg.batch_size
        self.device = device
        self.forward = forward
        self.pooling = cfg.pooling
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )

    @torch.no_grad()
    def encode(self, sequences):
        embeds_list = []
        num_batches = int(math.ceil(len(sequences) / self.batch_size))
        for batch in tqdm(
            batched(sequences, self.batch_size), total=num_batches, desc="Encoding..."
        ):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True).to(
                self.device
            )
            outputs = self.forward(**tokens)[0]

            mask = tokens.attention_mask.unsqueeze(-1)
            if self.pooling == "class":
                embeddings = outputs[:, 0, :]
            elif self.pooling == "mean":
                embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
            elif self.pooling == "max":
                mask_expanded = mask.expand(outputs.size())
                outputs[mask_expanded == 0] = -1e9
                embeddings, _ = outputs.max(dim=1)
            else:
                raise ValueError(f"self.pooling got unexpected value {self.pooling}")

            batch_embeds = nn.functional.normalize(embeddings, dim=1)
            embeds_list.append(batch_embeds)

        embeddings = torch.cat(embeds_list, dim=0)
        return embeddings


class SourMashEncoder(BaseEncoder):
    def __init__(self, cfg: SourMashConfig):
        """
        Constructor for encoder

        Args:
            k (int): K-mer size.
            scaled (int): Compression factor. 1000 means 1 hash kept per 1000 k-mers.
            moltype (str): 'DNA' or 'protein'.
        """
        self.k = cfg.k
        self.scaled = cfg.scaled

    def encode(self, sequences):
        """
        Takes a list of strings corresponding to sequences, and encodes their sourmash hash representations.
        Returns a list of SourmashSignature objects.
        """
        signatures = []

        for i, seq in tqdm(
            enumerate(sequences), total=len(sequences), desc="Hashing..."
        ):
            # Create a MinHash object
            # track_abundance=False is standard for simple search
            mh = MinHash(
                n=0,
                ksize=self.k,
                scaled=self.scaled,
                is_protein=False,
                track_abundance=False,
            )

            # Add sequence to the MinHash
            # Sourmash requires bytes or string depending on version, usually handles string fine in v4+
            mh.add_sequence(seq, force=True)

            # Wrap in a Signature.
            # We assign a temporary name; the Indexer will overwrite this with the real ID.
            sig = SourmashSignature(mh, name=str(i))
            signatures.append(sig)

        return signatures
