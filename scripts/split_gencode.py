import random
from pathlib import Path

from Bio import SeqIO
from jsonargparse import auto_cli
from tqdm import tqdm
from transformers import AutoTokenizer


def partition_randomly(input_list, n1, n2, n3):
    # 1. Validation check
    if n1 + n2 + n3 != len(input_list):
        raise ValueError(
            f"Sum of sizes ({n1 + n2 + n3}) does not match list length ({len(input_list)})"
        )

    random.shuffle(input_list)

    sublist1 = input_list[:n1]
    sublist2 = input_list[n1 : n1 + n2]
    sublist3 = input_list[n1 + n2 :]

    return sublist1, sublist2, sublist3


def filter_long_sequences(records, max_seq_len, tokenizer):
    print(f"Filtering sequences less than {max_seq_len}")
    long_records = []
    short_records = []
    for seq in tqdm(records):
        inputs = tokenizer(str(seq.seq), return_tensors="pt")
        seq_len = inputs.input_ids.shape[1]
        if seq_len < max_seq_len:
            short_records.append(seq)
        else:
            long_records.append(seq)
    return short_records, long_records


def main(
    gencode_fasta: str,
    train_split: float = 0.9,
    random_seed: int = 1337,
    max_seq_len: int = -1,
):
    random.seed(random_seed)
    gencode_fasta = Path(gencode_fasta)
    assert gencode_fasta.is_file()
    records = list(SeqIO.parse(gencode_fasta, "fasta"))
    if max_seq_len > 0:
        tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        records, long_records = filter_long_sequences(records, max_seq_len, tokenizer)
        long_path = (
            gencode_fasta.parent
            / f"{gencode_fasta.stem}.long.{max_seq_len}{gencode_fasta.suffix}"
        )
        SeqIO.write(long_records, long_path.resolve(), "fasta")

    print("Generating train/test/val split")
    N = len(records)
    N_train = int(train_split * N)
    N_test = int((N - N_train) / 2)
    N_val = N - N_train - N_test

    train_records, test_records, val_records = partition_randomly(
        records, N_train, N_test, N_val
    )

    for name, record in [
        ("train", train_records),
        ("test", test_records),
        ("val", val_records),
    ]:
        print(f"Writing {name} output record")
        split_path = (
            gencode_fasta.parent / f"{gencode_fasta.stem}.{name}{gencode_fasta.suffix}"
        )
        SeqIO.write(record, split_path.resolve(), "fasta")


if __name__ == "__main__":
    auto_cli(main)
