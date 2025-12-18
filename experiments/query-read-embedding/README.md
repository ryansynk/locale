# Mini SRA Search Experiment

## Embedding vectors
First, embed all read vectors for both models. This will take a long time.
```
python
python
```

## Searching queries over embeddings
Then, search over these vectors using the query set with `bash train.sh`, or individually for rawbert:
```
python search.py rawbert \
    --dataset_path="../../../rawbert_data/test_dataset.jsonl" \
    --accessions_path="../../../rawbert_data/accessions.csv" \
    --embeddings_bin_path="/fs/nexus-projects/sra_search/rawbert_embds.bin" \
    --embeddings_idx_map_path="/fs/nexus-projects/sra_search/rawbert_sra_id_map.parquet" \
    --metadata_path="/fs/nexus-scratch/ryansynk/rawbert_data/logan-seqstats.parquet" \
    --checkpoint_path=../../checkpoints/best_checkpoint.bsize128.lr4e6.pth.tar \
    --dim=128 \
    --K=131072 \
    --output="rawbert_k_10.parquet"
```
and DNAbert:
```
python search.py dnabert \
    --dataset_path="../../../rawbert_data/test_dataset.jsonl" \
    --accessions_path="../../../rawbert_data/accessions.csv" \
    --embeddings_bin_path="/fs/nexus-projects/sra_search/dnabert_embds.bin" \
    --embeddings_idx_map_path="/fs/nexus-projects/sra_search/dnabert_sra_id_map.parquet" \
    --metadata_path="/fs/nexus-scratch/ryansynk/rawbert_data/logan-seqstats.parquet" \
    --output="dnabert_k_10.parquet"
```

This will give you two output files, `rawbert_k_10.parquet` and `dnabert_k_10.parquet`

## Analyzing data 

