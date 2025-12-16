import numpy as np
import os
import polars as pl
from Bio import SeqIO
from jsonargparse import auto_cli
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import io
import math
import zstandard as zstd

from typing import Literal
import torch
from rawbert import RawBERT
from rawbert.utils.patch import patch_with_flash_lib


def count_reads(accessions_path, metadata_path):
    accs = pl.read_csv(accessions_path)
    accs = accs.sort(by="accession")
    metadata = pl.read_parquet(metadata_path)
    total_reads = accs.join(metadata, on="accession").select(pl.sum("seqstats_unitigs_nbseq")).item() 
    return total_reads

def get_model_and_tokenizer(model_str: str, checkpoint_path: str, dim: int, K: int):
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
        model.encode = lambda x: model(**x)[1]
    else:
        raise ValueError(f"Expected rawbert or dnabert for model, got {model}")

    return model, tokenizer


def embed_query_transcripts(model, tokenizer, dataset):
    vectors = []
    idx_mapping = []
    print("Encoding queries...")
    with torch.no_grad():
        for row in tqdm(
            dataset.iter_rows(named=True), desc="Encoding queries...", total=len(dataset)
        ):
            id = row["transcript_id"]
            seq = row["seq"]
            inputs = tokenizer(seq, return_tensors="pt").to(model.device)
            output = model.encode(inputs).cpu()
            vectors.append(output)
            idx_mapping.append((id, seq))

    vectors = torch.cat(vectors).numpy().astype(np.float32)
    return vectors, idx_mapping


def search_vectors(queries, embeddings_path, num_vectors, dim, batch_size, k):
    # 1. Memory map the large file (Instant, consumes no RAM)
    # Ensure your binary file is purely the vectors (no headers).
    # If there is a header, use the 'offset' parameter.
    X_disk = np.memmap(
        embeddings_path, dtype="float32", mode="r", shape=(num_vectors, dim)
    )

    # Initialize storage for top-k candidates
    # We will maintain a global list of top results
    global_top_dists = np.full((len(queries), k), -np.inf, dtype="float32")
    global_top_indices = np.full((len(queries), k), -1, dtype="int64")

    print(f"Starting scan over {num_vectors} vectors...")

    # 2. Iterate in chunks
    num_batches = math.ceil(num_vectors / batch_size)
    for i in tqdm(range(0, num_vectors, batch_size), desc="Searching index", total=num_batches):
        # Determine actual batch end (handle last chunk)
        end = min(i + batch_size, num_vectors)

        # Load chunk into RAM (This triggers the disk read)
        print("Loading chunk into RAM...")
        chunk = X_disk[i:end]

        # 3. Perform Matrix Multiplication (Inner Product)
        # If you need L2 distance, see note below*
        print("Performing dot product...")
        dists = np.dot(chunk, queries.T).T  # Shape: (num_queries, batch_size)

        # 4. Update Top-K
        # We concatenate current best with new batch results and sort
        # This logic is efficient because k is usually small

        # Combine current batch results with previous bests
        print("Updating topk...")
        combined_dists = np.concatenate([global_top_dists, dists], axis=1)
        combined_indices = np.concatenate(
            [
                global_top_indices,
                np.arange(i, end)
                + np.zeros((len(queries), 1), dtype="int64"),  # broadcast indices
            ],
            axis=1,
        )

        # Argpartition is faster than sort for finding top k
        # We want largest dot products (closest)
        top_k_idx_in_combined = np.argpartition(combined_dists, -k, axis=1)[:, -k:]

        # Gather the results
        rows = np.arange(len(queries))[:, None]
        global_top_dists = combined_dists[rows, top_k_idx_in_combined]
        global_top_indices = combined_indices[rows, top_k_idx_in_combined]

        # Optional: sort the final k for tidiness (argpartition is not sorted)
        sorted_order = np.argsort(global_top_dists, axis=1)[:, ::-1]
        global_top_dists = global_top_dists[rows, sorted_order]
        global_top_indices = global_top_indices[rows, sorted_order]

        if i % (batch_size * 10) == 0:
            print(f"Processed {end} / {num_vectors} vectors")

        print("Search complete.")

    return global_top_dists, global_top_indices

def iterate_batches_to_gpu(data_np, batch_size, device):
    num_vecs = data_np.shape[0]
    
    for i in range(0, num_vecs, batch_size):
        # Slice handles the tail end (non-divisible batch) automatically
        # e.g., data_np[299_999_000 : 300_000_100] returns just the last 1k rows
        batch_cpu = torch.from_numpy(data_np[i : i + batch_size])
        
        # Move to GPU
        # non_blocking=True allows overlap if you are doing compute asynchronously
        batch_gpu = batch_cpu.to(device, non_blocking=True)
        
        yield batch_gpu

def search_vectors_new(queries, embeddings_path, num_vectors, dim, batch_size, k, device):
    # queries (num_queries, D)

    # 1. Memory map the large file (Instant, consumes no RAM)
    # Ensure your binary file is purely the vectors (no headers).
    # If there is a header, use the 'offset' parameter.
    #X_disk = np.memmap(
    #    embeddings_path, dtype="float32", mode="r", shape=(num_vectors, dim)
    #)
    X_disk = np.memmap(
        embeddings_path, dtype="float32", mode="r", shape=(num_vectors, dim)
    )

    # Initialize storage for top-k candidates
    # We will maintain a global list of top results
    global_top_dists = np.full((len(queries), k), -np.inf, dtype="float32")
    global_top_indices = np.full((len(queries), k), -1, dtype="int64")

    print(f"Starting scan over {queries.shape[0]} queries...")
    for query in tqdm(queries, desc="Searching over queries", total=queries.shape[0]):
        # 3. Perform Matrix Multiplication (Inner Product)
        # If you need L2 distance, see note below*
        print("Performing dot product...")
        torch_query = torch.from_numpy(query).to(device)
        all_dists = []
        for batch in tqdm(iterate_batches_to_gpu(X_disk, batch_size, device), desc="Chunking", total=math.ceil(num_vectors / batch_size)):
            dists = (batch @ torch_query.T).T
            all_dists.append(dists)
        dists = torch.cat(dists).cpu().numpy()
        breakpoint()

        # 4. Update Top-K
        # We concatenate current best with new batch results and sort
        # This logic is efficient because k is usually small

        # Combine current batch results with previous bests
        print("Updating topk...")
        combined_dists = np.concatenate([global_top_dists, dists], axis=1)
        combined_indices = np.concatenate(
            [
                global_top_indices,
                np.arange(i, end)
                + np.zeros((len(queries), 1), dtype="int64"),  # broadcast indices
            ],
            axis=1,
        )

        # Argpartition is faster than sort for finding top k
        # We want largest dot products (closest)
        top_k_idx_in_combined = np.argpartition(combined_dists, -k, axis=1)[:, -k:]

        # Gather the results
        rows = np.arange(len(queries))[:, None]
        global_top_dists = combined_dists[rows, top_k_idx_in_combined]
        global_top_indices = combined_indices[rows, top_k_idx_in_combined]

        # Optional: sort the final k for tidiness (argpartition is not sorted)
        sorted_order = np.argsort(global_top_dists, axis=1)[:, ::-1]
        global_top_dists = global_top_dists[rows, sorted_order]
        global_top_indices = global_top_indices[rows, sorted_order]

        if i % (batch_size * 10) == 0:
            print(f"Processed {end} / {num_vectors} vectors")

        print("Search complete.")

    return global_top_dists, global_top_indices


def get_reads_from_indices(result_ids, result_dists, query_idx_mapping, embeddings_idx_map_path):
    df_ranges = pl.from_parquet(embeddings_idx_map_path)
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
                pl.col("dists")
            ]
        )
        .sort("row_nr")  # Restore original row order
        .drop("row_nr")
    )

    return result_df

def get_recall_precision(expected_df, actual_df):
    breakpoint()

def main(
    dataset_path: str,
    accessions_path: str,
    embeddings_bin_path: str,
    embeddings_idx_map_path: str,
    model_str: Literal["rawbert", "dnabert"],
    num_vectors: int = None,
    checkpoint_path: str = None,
    metadata_path: str = None,
    dim: int = None,
    K: int = None,
    batch_size: int = 10_000_000,
    top_k: int = 10,
):
    # Configuration
    if num_vectors is None:
        num_vectors = count_reads(accessions_path, metadata_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    model, tokenizer = get_model_and_tokenizer(model_str, checkpoint_path, dim, K)
    model = model.to(device)

    dataset = pl.read_ndjson(dataset_path)
    query_embeds, query_idx_mapping = embed_query_transcripts(model, tokenizer, dataset)

    dists, indices = search_vectors_new(
        query_embeds, embeddings_bin_path, num_vectors, dim, batch_size, top_k, device
    )
    reads_df = get_reads_from_indices(indices, dists, query_idx_mapping, embeddings_idx_map_path)
    recall, precision = get_recall_precision(dataset, reads_df)


if __name__ == "__main__":
    auto_cli(main)
