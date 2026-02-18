import io
import math
from itertools import islice
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zstandard as zstd
from Bio import SeqIO
from jsonargparse import auto_cli
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from rawbert import RawBERT
from rawbert.utils.patch import patch_with_flash_lib


def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def embed_accs(acc_paths, tokenizer, model, device, outfile, batch_size):
    model = model.to(device)
    print(f"Embedding {len(acc_paths)} accessions")
    dctx = zstd.ZstdDecompressor()
    global_idx = 0
    range_data = []

    with open(outfile, "wb") as f_out:
        for acc_path in acc_paths:
            acc_id = acc_path.name.split(".")[0]
            print(f"Processing {acc_id}")
            with open(acc_path, "rb") as compressed_file:
                # 1. Create a stream reader for the zstd data
                with dctx.stream_reader(compressed_file) as reader:
                    # 2. Wrap the byte stream in TextIOWrapper
                    # Biopython expects text (strings), but zstd outputs bytes.
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    records = list(SeqIO.parse(text_stream, "fasta"))

                    # Store mapping of linear index to accession
                    num_reads = len(records)
                    range_data.append({"start_id": global_idx, "accession": acc_id})

                    # 3. Pass the stream directly to Biopython
                    with torch.no_grad():
                        all_embeds = []
                        total_batches = math.ceil(num_reads / batch_size)
                        for batch in tqdm(
                            batched(records, batch_size), total=total_batches
                        ):
                            batch = list(batch)
                            batch_seqs = [str(record.seq) for record in batch]
                            batch_tokens = tokenizer(
                                batch_seqs, return_tensors="pt", padding=True
                            ).to(device)
                            embeds = model.encode(batch_tokens)
                            all_embeds.append(embeds)

                        all_embeds = torch.cat(all_embeds)
                        all_embeds = all_embeds.to("cpu").numpy().astype(np.float32)
                        f_out.write(all_embeds.tobytes())
                        f_out.flush()

                    global_idx += num_reads

    idfile = (outfile.parent / f"{outfile.stem}_sra_id_map.parquet").resolve()
    df_ranges = pl.DataFrame(range_data)
    df_ranges.write_parquet(idfile)


def main(
    test_dataset: str,
    accessions_path: str,
    reads_type: Literal["unitigs", "contigs"],
    output_bin: str,
    model_str: Literal["rawbert", "dnabert"],
    checkpoint_path: str = None,
    batch_size: int = 16384,
    dim: int = 128,
    K: int = 131072,
):
    df = pl.read_ndjson(test_dataset)
    accs = df.explode("accession_list")["accession_list"].unique().sort().to_list()
    acc_paths = [Path(accessions_path) / f"{acc}.{reads_type}.fa.zst" for acc in accs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    output_bin: Path = Path(output_bin).resolve()
    assert output_bin.parent.exists(), (
        f"Provided output dir = {output_bin.parent} does not exist!"
    )
    if model_str == "rawbert":
        print("Loading rawbert checkpoint...")
        model = RawBERT(dim=dim, K=K)
        assert checkpoint_path, "No checkpoint_path provided!"
        checkpoint_path: Path = Path(checkpoint_path).resolve()
        assert checkpoint_path.is_file()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        print("Loaded rawbert checkpoint successfully")
    elif model_str == "dnabert":
        model = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        patch_with_flash_lib(model)
        model = model.eval()
        model = model.to(device)
        model.encode = lambda x: F.normalize(model(**x)[1], dim=1)
    else:
        raise ValueError(f"Expected rawbert or dnabert for model, got {model}")

    embed_accs(acc_paths, tokenizer, model, device, output_bin, batch_size)


if __name__ == "__main__":
    auto_cli(main)
