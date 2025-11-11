from pathlib import Path
from Bio import SeqIO
from jsonargparse import auto_cli

def main(gencode_fasta: str, n: int):
    gencode_fasta = Path(gencode_fasta)
    assert gencode_fasta.is_file()
    records = list(SeqIO.parse(gencode_fasta, "fasta"))
    k, m = divmod(len(records), n)
    records_split = [records[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
    for i, split in enumerate(records_split):
        split_path = gencode_fasta.parent / f"{gencode_fasta.stem}.{i}{gencode_fasta.suffix}"
        SeqIO.write(split, split_path.resolve(), "fasta")


if __name__=="__main__":
    auto_cli(main)