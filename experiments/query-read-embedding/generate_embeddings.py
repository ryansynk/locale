import io
import itertools
import math
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
import zstandard as zstd
from Bio import SeqIO
from jsonargparse import auto_cli
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from rawbert import RawBERT
from rawbert.utils.patch import patch_with_flash_lib


def embed_accs(accs, tokenizer, model, model_str, device, outpath, batch_size):
    model = model.to(device)
    print(f"Embedding {len(accs)} accessions")
    dctx = zstd.ZstdDecompressor()
    global_idx = 0
    range_data = []

    outfile = outpath / f"{model_str}_embds.bin"
    with open(outfile, "wb") as f_out:
        for acc_row in accs.iter_rows(named=True):
            acc_id = acc_row["accession"]
            acc_path = acc_row["path"]
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
                            itertools.batched(records, batch_size), total=total_batches
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

    idfile = outpath / f"{model_str}_sra_id_map.parquet"
    df_ranges = pl.DataFrame(range_data)
    df_ranges.write_parquet(idfile)


def main(
    accessions_csv: str,
    output_dir: str,
    model_str: Literal["rawbert", "dnabert"],
    checkpoint_path: str = None,
    batch_size: int = 16384,
    dim: int = 128,
    K: int = 131072,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    output_dir = Path(output_dir).resolve()
    assert output_dir.exists(), f"Provided output dir = {output_dir} does not exist!"
    if model_str == "rawbert":
        print("Loading rawbert checkpoint...")
        model = RawBERT(dim=dim, K=K)
        assert checkpoint_path, "No checkpoint_path provided!"
        checkpoint_path = Path(checkpoint_path).resolve()
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
        model.encode = lambda x: model(**x)[1]
    else:
        raise ValueError(f"Expected rawbert or dnabert for model, got {model}")

    accs = pl.read_csv(accessions_csv)
    accs = accs.sort(by="accession")
    embed_accs(accs, tokenizer, model, model_str, device, output_dir, batch_size)


if __name__ == "__main__":
    auto_cli(main)
