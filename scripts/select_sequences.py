import argparse
import pathlib
import os
import random
from Bio import SeqIO
from jsonargparse import auto_cli


def format_number(num):
    if num % 1000 == 0 and num > 0:
        return f"{num // 1000}k"
    return str(num)


def select_sequences(transcriptome_fasta, N, L_min, L_max):
    """
    Selects N distinct transcript sequences from a FASTA file,
    where each sequence length is between L_min and L_max.
    """
    transcripts = list(SeqIO.parse(transcriptome_fasta, "fasta"))
    filtered_transcripts = [t for t in transcripts if L_min <= len(t) <= L_max]
    assert N <= len(filtered_transcripts)
    selected = random.sample(filtered_transcripts, N)
    return [(t.id, str(t.seq)) for t in selected]


def main(
    transcriptome_fasta: str,
    N_train: int = 90000,
    N_val: int = 5000,
    N_test: int = 5000,
    L_min: int = 1000,
    L_max: int = 10000,
):
    random.seed(1337)

    if not os.path.exists(transcriptome_fasta):
        print(f"Error: File not found at {transcriptome_fasta}")
        return
    N_total = N_train + N_val + N_test
    sequences = select_sequences(transcriptome_fasta, N_total, L_min, L_max)

    data_dir = pathlib.Path(__file__).parent.resolve() / ".." / "data" / "dataset"
    data_dir.mkdir(exist_ok=True)



    #for split, selected_sequences in [
    #    ("train", train_sequences),
    #    ("val", val_sequences),
    #    ("test", test_sequences),
    #]:
        split_dir = data_dir / split
        split_dir.mkdir(exist_ok=True)
        for idx, record in selected_sequences:
            assert record.seq.count("N") == 0
            seq_dir = split_dir / f"{idx}"
            seq_dir.mkdir(exist_ok=True)
            output_file = seq_dir / "query.fa"
            with output_file.open("w") as f:
                SeqIO.write(record, f, "fasta")

    print(
        f"Successfully selected {N_total} sequences and saved them to {str(data_dir)}"
    )


if __name__ == "__main__":
    auto_cli(main)
