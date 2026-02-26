from pathlib import Path
import polars as pl
from torch.utils.data import Dataset
from pyfaidx import Fasta


class BadreadPairDataset(Dataset):
    # def __init__(self, reads_fastq, reference_fasta, tokenizer, max_len=1024):
    def __init__(
        self, dataset_path, reference_fasta_path, augment_config, num_examples=None
    ):
        self.df = pl.read_parquet(dataset_path).filter(
            pl.col("length") <= augment_config.max_seq_len
        )
        # self.df = pl.read_parquet(dataset_path).filter(
        #    (pl.col("strand") == "+") & (pl.col("length") <= augment_config.max_seq_len)
        # )

        reference_fasta_path: Path = Path(reference_fasta_path).resolve()
        self.reference = Fasta(reference_fasta_path)

    def __len__(self):
        return len(self.reads)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        ref_seq = str(
            self.reference[row["chromosome"]][
                row["reference_start"] : row["reference_end"]
            ]
        )

        # Extract reference segment
        if row["strand"] == "-":
            ref_seq = reverse_complement(ref_seq)


def reverse_complement(seq):
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]
