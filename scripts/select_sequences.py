import argparse
import pathlib
import os
import random
from Bio import SeqIO

def format_number(num):
    if num % 1000 == 0 and num > 0:
        return f"{num // 1000}k"
    return str(num)

def select_sequences(fasta_file, n, l):
    """
    Selects N non-overlapping sequences of length L from a FASTA file.
    """
    transcripts = list(SeqIO.parse(fasta_file, "fasta"))

    selected_sequences = []
    selected_transcript_ids = []
    while len(selected_sequences) < n:
        random_transcript_idx = random.randint(0, len(transcripts))
        if random_transcript_idx in selected_transcript_ids:
            continue
        selected_transcript_ids.append(random_transcript_idx)
        sequence = transcripts[random_transcript_idx]
        if len(sequence) >= l:
            selected_sequences.append((random_transcript_idx, sequence))
            
    return selected_sequences

def main():
    parser = argparse.ArgumentParser(description="Select non-overlapping sequences from a FASTA file.")
    parser.add_argument('--N_train', type=int, default=10000, help="Number of sequences to select for train dataset.")
    parser.add_argument('--N_val', type=int, default=2000, help="Number of sequences to select for val dataset.")
    parser.add_argument('--N_test', type=int, default=2000, help="Number of sequences to select for test dataset.")
    parser.add_argument('--L', type=int, default=1000, help="Length of each sequence.")
    parser.add_argument('fasta_file', type=str, help="Path to the FASTA file.")
    args = parser.parse_args()

    n_train = args.N_train
    n_val = args.N_val
    n_test = args.N_test
    n_total = n_train + n_val + n_test
    l = args.L
    fasta_file = args.fasta_file
    random.seed(1337)

    if not os.path.exists(fasta_file):
        print(f"Error: File not found at {fasta_file}")
        return

    sequences = select_sequences(fasta_file, n_total, l)
    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train:(n_train + n_val)]
    test_sequences = sequences[(n_train + n_val):]

    data_dir = pathlib.Path(__file__).parent.resolve() / ".." / "data" / "dataset"
    data_dir.mkdir(exist_ok=True)

    for split, selected_sequences in [("train", train_sequences), ("val", val_sequences), ("test", test_sequences)]:
        split_dir = data_dir / split
        split_dir.mkdir(exist_ok=True)
        for idx, record in selected_sequences:
            assert record.seq.count("N") == 0
            seq_dir = split_dir / f"{idx}"
            seq_dir.mkdir(exist_ok=True)
            output_file = seq_dir / "query.fa"
            with output_file.open("w") as f:
                SeqIO.write(record, f, "fasta")

    print(f"Successfully selected {n_total} sequences and saved them to {str(data_dir)}")

if __name__ == "__main__":
    main()
