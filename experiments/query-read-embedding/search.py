import math
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

from rawbert import RawBERT
from rawbert.utils.patch import patch_with_flash_lib


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


def count_reads(accessions_path, metadata_path):
    accs = pl.read_csv(accessions_path)
    accs = accs.sort(by="accession")
    metadata = pl.read_parquet(metadata_path)
    total_reads = (
        accs.join(metadata, on="accession")
        .select(pl.sum("seqstats_unitigs_nbseq"))
        .item()
    )
    return total_reads


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
            output = model.encode(inputs).cpu()
            vectors.append(output)
            idx_mapping.append((id, seq))

    vectors = torch.cat(vectors).numpy().astype(np.float32)
    return vectors, idx_mapping


def search_vectors(queries, embeddings_path, num_vectors, dim, batch_size, k, device):
    # 1. Memory map the large file (Instant, consumes no RAM)
    # Ensure your binary file is purely the vectors (no headers).
    # If there is a header, use the 'offset' parameter.
    X_disk = np.memmap(
        embeddings_path, dtype="float32", mode="r", shape=(num_vectors, dim)
    )
    num_queries: int = queries.shape[0]
    # Initialize Global Buffers to store the best results found so far
    # Values initialized to -infinity, Indices to -1
    global_topk_vals: torch.Tensor = torch.full(
        (num_queries, k), float("-inf"), device=device
    )
    global_topk_indices: torch.Tensor = torch.full(
        (num_queries, k), -1, dtype=torch.long, device=device
    )

    print(f"Starting scan over {num_vectors} vectors...")
    queries = torch.from_numpy(queries).to(device)
    # 2. Iterate in chunks
    num_batches = math.ceil(num_vectors / batch_size)
    current_offset = 0
    for i in tqdm(
        range(0, num_vectors, batch_size), desc="Searching index", total=num_batches
    ):
        # Determine actual batch end (handle last chunk)
        end = min(i + batch_size, num_vectors)

        # Load chunk into RAM (This triggers the disk read)
        chunk = torch.from_numpy(X_disk[i:end]).to(device)

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

    if i % (batch_size * 10) == 0:
        print(f"Processed {end} / {num_vectors} vectors")

    print("Search complete.")

    return global_topk_vals.cpu().numpy(), global_topk_indices.cpu().numpy()


def search_vectors_pool_accession(
    queries,
    embeddings_path,
    embeddings_idx_map_path,
    num_vectors,
    dim,
    batch_size,
    k,
    device,
    query_idx_mapping,
):
    # 1. Memory map the large file (Instant, consumes no RAM)
    # Ensure your binary file is purely the vectors (no headers).
    # If there is a header, use the 'offset' parameter.
    X_disk = np.memmap(
        embeddings_path, dtype="float32", mode="r", shape=(num_vectors, dim)
    )
    df_offsets = pl.read_parquet(embeddings_idx_map_path)
    num_queries: int = queries.shape[0]

    print(f"Starting scan over {num_vectors} vectors...")
    queries = torch.from_numpy(queries).to(device)

    accs = df_offsets["accession"].to_list()
    start_indices = df_offsets["start_id"].to_list()
    results = {}
    query_ids = [id for (id, seq) in query_idx_mapping]
    for id in query_ids:
        results[id] = {}

    current_offset = 0
    k_pool = 100
    for start_idx, end_idx, acc in tqdm(
        list(zip(start_indices, start_indices[1:] + [num_vectors], accs))
    ):
        global_topk_vals: torch.Tensor = torch.full(
            (num_queries, k_pool), float("-inf"), device=device
        )
        global_topk_indices: torch.Tensor = torch.full(
            (num_queries, k_pool), -1, dtype=torch.long, device=device
        )
        for i in range(start_idx, end_idx, batch_size):
            # Determine actual batch end (handle last chunk)
            # end = min(i + batch_size, num_vectors_in_acc)
            end = min(i + batch_size, end_idx)

            # Load chunk into RAM (This triggers the disk read)
            chunk = torch.from_numpy(X_disk[i:end]).to(device)

            # Perform Matrix Multiplication (Inner Product)
            # chunk is batch_size x dim, queries is num_queries x dim
            # dists is num_queries x batch_size
            dists = (chunk @ queries.T).T

            local_k = min(k_pool, batch_size)
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
            global_topk_vals, best_of_both_idx = torch.topk(
                combined_vals, k_pool, dim=1
            )
            global_topk_indices = torch.gather(combined_indices, 1, best_of_both_idx)

            current_offset += batch_size
            del chunk, dists, local_dists, local_indices

        for id, val in zip(query_ids, global_topk_vals.sum(dim=1).cpu().tolist()):
            results[id].update({acc: val})

    print("Search complete.")
    filtered_results = []
    for transcript_id in results.keys():
        # list of (acc, score)
        sorted_accs = sorted(
            results[transcript_id].items(), key=lambda f: f[1], reverse=True
        )
        filtered_results.append(
            {
                "transcript_id": transcript_id,
                "accession": [acc for acc, score in sorted_accs[:k]],
            }
        )

    results_df = pl.from_dicts(filtered_results)
    return results_df


def get_accs_from_indices(
    result_ids, result_dists, query_idx_mapping, embeddings_idx_map_path
):
    print("Mapping results to accessions")
    df_ranges = pl.read_parquet(embeddings_idx_map_path)
    # Create a DataFrame for your queries
    # Note: sort("query_id") is usually required for efficient asof joins
    df_dists = pl.from_numpy(result_dists, schema={"dists": pl.List(pl.Float32)})
    df_queries = pl.from_numpy(result_ids, schema={"query_ids": pl.List(pl.Int64)})
    df_queries = pl.concat([df_queries, df_dists], how="horizontal")
    df_queries = df_queries.with_columns(
        pl.Series(name="transcript_id", values=[id for (id, seq) in query_idx_mapping])
    )
    df_queries = (
        df_queries.with_row_index("row_nr")  # Create a unique ID for the original row
        .explode("query_ids")  # Flatten the list: 1 row per ID
        .sort("query_ids")  # Sort is required for join_asof
    )
    # Perform the lookup
    # 'backward' strategy means: find the closest start_id <= query_id
    df_mapped = df_queries.join_asof(
        df_ranges, left_on="query_ids", right_on="start_id", strategy="backward"
    )

    # Calculate the local read ID (Read index within that specific accession)
    df_mapped = df_mapped.with_columns(
        (pl.col("query_ids") - pl.col("start_id")).alias("read_offset")
    )

    result_df = (
        df_mapped.group_by("row_nr", maintain_order=True)
        .agg(
            [
                pl.first("transcript_id"),  # Retain original metadata
                pl.col("query_ids"),  # The original IDs (now sorted)
                pl.col("accession"),  # The mapped accessions (as a list)
                pl.col("read_offset"),  # The local offsets (as a list)
                pl.col("dists"),
            ]
        )
        .sort("row_nr")  # Restore original row order
        .drop("row_nr")
    )

    return result_df


def main(
    model,
    tokenizer,
    dim: int,
    dataset_path: str,
    accessions_path: str,
    embeddings_bin_path: str,
    embeddings_idx_map_path: str,
    metadata_path: str,
    batch_size: int,
    topk: int,
    output_parquet: str,
    search_strategy: Literal["standard", "by-accession"],
):
    warnings.filterwarnings("ignore", message=".*Increasing alibi size.*")
    warnings.filterwarnings("ignore", message=".*Unable to import Triton.*")
    transformers_logging.set_verbosity_error()

    # Configuration
    num_vectors = count_reads(accessions_path, metadata_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "No GPU available, aborting"
    model = model.to(device)

    dataset = pl.read_ndjson(dataset_path)
    query_embeds, query_idx_mapping = embed_query_transcripts(model, tokenizer, dataset)

    if search_strategy == "standard":
        dists, indices = search_vectors(
            query_embeds,
            embeddings_bin_path,
            num_vectors,
            dim,
            batch_size,
            topk,
            device,
        )
        results_df = get_accs_from_indices(
            indices, dists, query_idx_mapping, embeddings_idx_map_path
        )
    elif search_strategy == "by-accession":
        results_df = search_vectors_pool_accession(
            query_embeds,
            embeddings_bin_path,
            embeddings_idx_map_path,
            num_vectors,
            dim,
            batch_size,
            topk,
            device,
            query_idx_mapping,
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
        "--embeddings_bin_path",
        type=str,
        required=True,
        help="Path to embeddings bin file",
    )
    shared_parser.add_argument(
        "--embeddings_idx_map_path",
        type=str,
        required=True,
        help="Path to idx-read mapping file",
    )
    shared_parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to logan seqstats metadata",
    )
    shared_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to CSV output file for search results",
    )
    shared_parser.add_argument(
        "--batch_size",
        type=int,
        default=1_000_000,
        help="Batch size to process embedding vecs with",
    )
    shared_parser.add_argument(
        "--topk", type=int, default=10, help="Number of k to search with"
    )
    shared_parser.add_argument(
        "--strategy", type=str, choices=["standard", "by-accession"], default="standard"
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
        args.embeddings_bin_path,
        args.embeddings_idx_map_path,
        args.metadata_path,
        args.batch_size,
        args.topk,
        args.output,
        args.strategy,
    )
