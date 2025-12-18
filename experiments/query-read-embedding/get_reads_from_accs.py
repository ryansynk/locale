import io
from pathlib import Path

import polars as pl
import zstandard as zstd
from Bio import SeqIO
from jsonargparse import auto_cli
from tqdm import tqdm


def get_reads_from_accs(df, logan_dir):
    print("Mapping accessions to read sequences")
    # ---------------------------------------------------------
    # STEP 1: PREPARE & EXPLODE
    # ---------------------------------------------------------
    # We add an index to 'offsets' so we can recover the exact order later.
    # If we don't do this, the reconstruction might shuffle the reads.
    # df_long = df.with_columns(
    #    pl.int_range(0, pl.col("accession").len()).alias("list_idx")
    # ).explode("accession", "read_offset", "list_idx")
    df_long = df.with_columns(
        pl.int_ranges(0, pl.col("accession").list.len()).alias("list_idx")
    ).explode("accession", "read_offset", "list_idx")

    # ---------------------------------------------------------
    # STEP 2: PARTITION & FETCH
    # ---------------------------------------------------------
    # We split the dataframe by accession. This groups all requests for "file_A" together.
    # We process each file once, extracting only the needed sequences.

    processed_chunks = []
    logan_dir = Path(logan_dir).resolve()
    dctx = zstd.ZstdDecompressor()
    # Partitioning by accession prevents reloading the same file multiple times
    num_accs = len(df_long["accession"].unique().to_list())
    for accession, group in tqdm(
        df_long.partition_by("accession", as_dict=True).items(),
        desc="Getting reads from acc files",
        total=num_accs,
    ):
        # A. Load the heavy data into memory ONCE for this accession
        acc_path = logan_dir / f"{accession[0]}.unitigs.fa.zst"
        with open(acc_path, "rb") as compressed_file:
            # 1. Create a stream reader for the zstd data
            with dctx.stream_reader(compressed_file) as reader:
                # 2. Wrap the byte stream in TextIOWrapper
                # Biopython expects text (strings), but zstd outputs bytes.
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                seqs = list(SeqIO.parse(text_stream, "fasta"))
                seqs = [str(seq.seq) for seq in seqs]

                # B. Convert to a lightweight dataframe for a vectorized join
                #    (If fasta is huge, you can optimize this to only create rows for needed indices)
                fasta_df = pl.DataFrame(
                    {"read_offset": range(len(seqs)), "sequence_read": seqs}
                ).with_columns(
                    pl.col("read_offset").cast(pl.Int64)
                )  # Ensure type match

                # Join the user's group with the actual sequences
                # We use 'offsets' as the join key.
                joined_chunk = group.join(fasta_df, on="read_offset", how="left")
                processed_chunks.append(joined_chunk)

    # ---------------------------------------------------------
    # STEP 3: RECONSTRUCT
    # ---------------------------------------------------------
    result = (
        pl.concat(processed_chunks)
        # Important: Sort by transcript and the list_idx we made earlier
        # to ensure [read_0, read_1] doesn't become [read_1, read_0]
        .sort("transcript_id", "list_idx")
        .group_by("transcript_id", maintain_order=True)
        .agg(
            [
                pl.col("accession"),
                pl.col("read_offset"),
                pl.col("sequence_read"),  # This is your new column
            ]
        )
    )
    return result


def main(results_csv: str, logan_dir: str):
    df = pl.read_csv(results_csv)
    df = get_reads_from_accs(df, logan_dir)
    df.write_csv(results_csv)


if __name__ == "__main__":
    auto_cli(main)
