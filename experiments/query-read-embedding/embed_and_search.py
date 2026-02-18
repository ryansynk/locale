import io
import math
import warnings
from argparse import ArgumentParser
from itertools import islice
from pathlib import Path

import polars as pl
import torch
import zstandard as zstd
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

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


def get_rawbert_model(args):
    model = RawBERT(dim=args.dim, K=args.K)
    checkpoint_path = Path(args.checkpoint_path).resolve()
    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    return model, tokenizer, args.dim


def get_dnabert_model(args):
    model = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    patch_with_flash_lib(model)
    model = model.eval()
    model.encode = lambda x: model(**x)[1]
    # TODO get this from model?
    dim = 768
    return model, tokenizer, dim


def embed_query_transcripts(model, tokenizer, dataset):
    vectors = []
    idx_mapping = []
    with torch.no_grad():
        for row in tqdm(
            dataset.iter_rows(named=True),
            desc="Encoding queries...",
            total=len(dataset),
        ):
            id = row["transcript_id"]
            seq = row["seq"]
            inputs = tokenizer(seq, return_tensors="pt").to(model.device)
            output = model.encode(inputs)
            vectors.append(output)
            idx_mapping.append((id, seq))

    vectors = torch.cat(vectors).to(torch.float32)
    return vectors, idx_mapping


def get_embeds_of_acc(num_reads, batch_size, records, device):
    with torch.no_grad():
        all_embeds = []
        total_batches = math.ceil(num_reads / batch_size)
        for batch in tqdm(
            batched(records, batch_size), total=total_batches, leave=False
        ):
            batch = list(batch)
            batch_seqs = [str(record.seq) for record in batch]
            batch_tokens = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(
                device
            )
            embeds = model.encode(batch_tokens)
            all_embeds.append(embeds)

        all_embeds = torch.cat(all_embeds).to(torch.float32)
    return all_embeds


def search_embeds_of_acc(
    queries,
    k,
    batch_size,
    device,
    embeds,
    acc,
    query_idx_mapping,
):
    num_queries = queries.shape[0]
    global_topk_vals: torch.Tensor = torch.full(
        (num_queries, k), float("-inf"), device=device
    )
    global_topk_indices: torch.Tensor = torch.full(
        (num_queries, k), -1, dtype=torch.long, device=device
    )
    num_reads = embeds.shape[0]
    current_offset = 0
    query_ids = [id for (id, seq) in query_idx_mapping]
    for i in range(0, num_reads, batch_size):
        # Determine actual batch end (handle last chunk)
        # end = min(i + batch_size, num_vectors_in_acc)
        end = min(i + batch_size, num_reads)

        chunk = embeds[i:end]
        # Perform Matrix Multiplication (Inner Product)
        # chunk is batch_size x dim, queries is num_queries x dim
        # dists is num_queries x batch_size
        dists = (chunk @ queries.T).T

        local_k = min(k, batch_size)

        # topk_dists, topk_indices are num_queries x local_k
        local_dists, local_indices = torch.topk(dists, local_k, dim=1)
        global_mapped_indices = local_indices + current_offset

        # 4. Merge with Global Buffer
        # Concatenate current global best with new local candidates
        combined_vals = torch.cat([global_topk_vals, local_dists], dim=1)
        combined_indices = torch.cat(
            [global_topk_indices, global_mapped_indices], dim=1
        )

        # 6. Reduce back to Top-K
        # This keeps our memory footprint constant regardless of total array size
        global_topk_vals, best_of_both_idx = torch.topk(combined_vals, k, dim=1)
        global_topk_indices = torch.gather(combined_indices, 1, best_of_both_idx)

        current_offset += batch_size
        del chunk, dists, local_dists, local_indices

    query_ids = [id for (id, seq) in query_idx_mapping]
    results = []
    for id, val in zip(query_ids, global_topk_vals.sum(dim=1).cpu().tolist()):
        results.append({"transcript_id": id, "accession": acc, "score": val})
    results = pl.from_dicts(results)
    return results


def embed_and_search_accs(
    queries,
    query_idx_mapping,
    accs,
    tokenizer,
    model,
    device,
    embedding_batch_size,
    search_batch_size,
    k,
):
    model = model.to(device)
    print(f"Embedding and searching over {len(accs)} accessions")
    dctx = zstd.ZstdDecompressor()
    results = []

    for acc_row in tqdm(
        accs.iter_rows(named=True), total=len(accs), desc="Processing accs..."
    ):
        acc_id = acc_row["accession"]
        acc_path = acc_row["path"]
        with open(acc_path, "rb") as compressed_file:
            # 1. Create a stream reader for the zstd data
            with dctx.stream_reader(compressed_file) as reader:
                # 2. Wrap the byte stream in TextIOWrapper
                # Biopython expects text (strings), but zstd outputs bytes.
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                records = list(SeqIO.parse(text_stream, "fasta"))

                # Store mapping of linear index to accession
                num_reads = len(records)
                tqdm.write(f"Embedding {num_reads} accession reads for {acc_id}")
                embeds = get_embeds_of_acc(
                    num_reads, embedding_batch_size, records, device
                )

                tqdm.write(f"Searching accession reads for {acc_id}")
                results_acc = search_embeds_of_acc(
                    queries,
                    k,
                    search_batch_size,
                    device,
                    embeds,
                    acc_id,
                    query_idx_mapping,
                )

                results.append(results_acc)

    results = pl.concat(results)
    # Apply topk cutoff
    results = results.group_by("transcript_id").agg(
        pl.col("accession").sort_by("score", descending=True).head(k),
        pl.col("score").sort_by("score", descending=True).head(k),
    )
    return results


def main(
    model,
    tokenizer,
    dim: int,
    dataset_path: str,
    accessions_path: str,
    embedding_batch_size: int,
    search_batch_size: int,
    topk: int,
    output_parquet: str,
):
    warnings.filterwarnings("ignore", message=".*Increasing alibi size.*")
    warnings.filterwarnings("ignore", message=".*Unable to import Triton.*")
    transformers_logging.set_verbosity_error()

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "No GPU available, aborting"
    model = model.to(device)

    accs = pl.read_csv(accessions_path)
    accs = accs.sort(by="accession")
    dataset = pl.read_ndjson(dataset_path)
    query_embeds, query_idx_mapping = embed_query_transcripts(model, tokenizer, dataset)

    results_df = embed_and_search_accs(
        query_embeds,
        query_idx_mapping,
        accs,
        tokenizer,
        model,
        device,
        embedding_batch_size,
        search_batch_size,
        topk,
    )
    results_df.write_parquet(output_parquet)


if __name__ == "__main__":
    # 1. Create the parent parser for shared arguments
    # add_help=False is CRITICAL here to avoid conflict
    shared_parser = ArgumentParser(add_help=False)
    shared_parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to test dataset jsonl"
    )
    shared_parser.add_argument(
        "--accessions_path", type=str, required=True, help="Path to accessions csv"
    )
    shared_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to parquet output file for search results",
    )
    shared_parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=16384,
        help="Batch size to embed vecs with",
    )
    shared_parser.add_argument(
        "--search_batch_size",
        type=int,
        default=1_000_000,
        help="Batch size to process embedding vecs with",
    )
    shared_parser.add_argument(
        "--topk", type=int, default=10, help="Number of k to search with"
    )

    # 2. Create the main top-level parser
    main_parser = ArgumentParser()
    subparsers = main_parser.add_subparsers(dest="model_name")

    # 3. Create subparsers that inherit from shared_parser
    # You can pass multiple parents in the list
    rawbert_parser = subparsers.add_parser(
        "rawbert", parents=[shared_parser], help="Benchmark rawbert model"
    )
    rawbert_parser.add_argument(
        "--checkpoint_path", type=str, help="Path to rawbert checkpoint"
    )
    rawbert_parser.add_argument(
        "--dim", type=int, help="Dimension of rawbert embedding"
    )
    rawbert_parser.add_argument("--K", type=int, help="Length of rawbert queue")
    rawbert_parser.set_defaults(func=get_rawbert_model)

    dnabert_parser = subparsers.add_parser(
        "dnabert", parents=[shared_parser], help="Benchmark dnabert model"
    )
    dnabert_parser.set_defaults(func=get_dnabert_model)
    args = main_parser.parse_args()
    model, tokenizer, dim = args.func(args)

    main(
        model,
        tokenizer,
        dim,
        args.dataset_path,
        args.accessions_path,
        args.embedding_batch_size,
        args.search_batch_size,
        args.topk,
        args.output,
    )
