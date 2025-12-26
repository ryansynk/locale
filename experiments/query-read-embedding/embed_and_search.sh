python embed_and_search.py rawbert \
    --dataset_path="../../../rawbert_data/test.jsonl" \
    --accessions_path="../../../rawbert_data/accessions.csv" \
    --embedding_batch_size=4096 \
    --search_batch_size=500000 \
    --checkpoint_path=../../checkpoints/best_checkpoint.bsize128.lr4e6.pth.tar \
    --dim=128 \
    --K=131072 \
    --output="rawbert_k_10_by_accession_full_dataset.parquet"