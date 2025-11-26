import random
from pathlib import Path

from Bio import SeqIO
from jsonargparse import auto_cli


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


def main(
    gencode_fasta: str,
    train_split: float = 0.9,
    random_seed: int = 1337,
):
    random.seed(random_seed)
    gencode_fasta = Path(gencode_fasta)
    assert gencode_fasta.is_file()
    records = list(SeqIO.parse(gencode_fasta, "fasta"))
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
        split_path = (
            gencode_fasta.parent / f"{gencode_fasta.stem}.{name}{gencode_fasta.suffix}"
        )
        SeqIO.write(record, split_path.resolve(), "fasta")


if __name__ == "__main__":
    auto_cli(main)
