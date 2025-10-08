import os
import argparse
import json
from Bio import SeqIO

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = get_args()
    base_directories = ['dataset/train', 'dataset/val', 'dataset/test']

    for base_dir in base_directories:
        name = base_dir.split("/")[-1]
        jsonl_path = os.path.join(args.data_dir, f"{name}.jsonl")
        base_dir = os.path.join(args.data_dir, base_dir)

        # Check if the base directory exists
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory '{base_dir}' not found. Skipping.")
            continue

        with open(jsonl_path, "a") as f:
            print(f"\n--- Processing directory: {base_dir} ---")
            for dirpath, _, filenames in os.walk(base_dir):
                # Check if 'query.fa' exists in the current directory
                query_check = ('query.fa' in filenames)
                read_check = ("paired_end_com1.fq" in filenames) and ("paired_end_com2.fq" in filenames)
                if query_check and read_check:
                    query_path = os.path.join(dirpath, "query.fa")
                    left_reads_path = os.path.join(dirpath, "paired_end_com1.fq")
                    right_reads_path = os.path.join(dirpath, "paired_end_com2.fq")
                    query = str(list(SeqIO.parse(query_path, "fasta"))[0].seq)
                    left_reads = list(SeqIO.parse(left_reads_path, "fastq"))
                    right_reads = list(SeqIO.parse(right_reads_path, "fastq"))

                    reads = []
                    for left, right in zip(left_reads, right_reads):
                        reads.append(str(left.seq))
                        reads.append(str(right.seq))
                    # write (query, reads) to jsonl
                    record = {"id": int(dirpath.split("/")[-1]), "query": query, "reads": reads}
                    json_record = json.dumps(record)
                    f.write(json_record + '\n')
                elif 'query.fa' in filenames or "paired_end_com1.fq" in filenames or "paired_end_com2.fq" in filenames:
                    raise ValueError(f"Files not found in {dirpath}")

if __name__=="__main__":
    main()
