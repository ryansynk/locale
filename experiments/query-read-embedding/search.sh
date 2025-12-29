python search.py rawbert \
    --dataset_path="../../../rawbert_data/small.test.jsonl" \
    --accessions_path="../../../rawbert_data/small.accessions.csv" \
    --embeddings_bin_path="/fs/nexus-projects/sra_search/rawbert_embeds_small.bin" \
    --embeddings_idx_map_path="/fs/nexus-projects/sra_search/rawbert_embeds_small_sra_id_map.parquet" \
    --checkpoint_path=../../checkpoints/best_checkpoint.bsize128.lr4e6.pth.tar \
    --dim=128 \
    --K=131072 \
    --output="rawbert_k_1_small.parquet" \
    --topk=1 \
    --strategy="standard" \

python search.py rawbert \
    --dataset_path="../../../rawbert_data/small.test.jsonl" \
    --accessions_path="../../../rawbert_data/small.accessions.csv" \
    --embeddings_bin_path="/fs/nexus-projects/sra_search/rawbert_embeds_small.bin" \
    --embeddings_idx_map_path="/fs/nexus-projects/sra_search/rawbert_embeds_small_sra_id_map.parquet" \
    --checkpoint_path=../../checkpoints/best_checkpoint.bsize128.lr4e6.pth.tar \
    --dim=128 \
    --K=131072 \
    --output="rawbert_k_5_small.parquet" \
    --topk=5 \
    --strategy="standard" \

python search.py rawbert \
    --dataset_path="../../../rawbert_data/small.test.jsonl" \
    --accessions_path="../../../rawbert_data/small.accessions.csv" \
    --embeddings_bin_path="/fs/nexus-projects/sra_search/rawbert_embeds_small.bin" \
    --embeddings_idx_map_path="/fs/nexus-projects/sra_search/rawbert_embeds_small_sra_id_map.parquet" \
    --checkpoint_path=../../checkpoints/best_checkpoint.bsize128.lr4e6.pth.tar \
    --dim=128 \
    --K=131072 \
    --output="rawbert_k_1_small_by_accession.parquet" \
    --topk=1 \
    --strategy="by-accession" \

python search.py dnabert \
    --dataset_path="../../../rawbert_data/small.test.jsonl" \
    --accessions_path="../../../rawbert_data/small.accessions.csv" \
    --embeddings_bin_path="/fs/nexus-projects/sra_search/dnabert_embeds_small.bin" \
    --embeddings_idx_map_path="/fs/nexus-projects/sra_search/dnabert_embeds_small_sra_id_map.parquet" \
    --output="dnabert_k_1_small.parquet" \
    --topk=1 \
    --strategy="standard" \

python search.py dnabert \
    --dataset_path="../../../rawbert_data/small.test.jsonl" \
    --accessions_path="../../../rawbert_data/small.accessions.csv" \
    --embeddings_bin_path="/fs/nexus-projects/sra_search/dnabert_embeds_small.bin" \
    --embeddings_idx_map_path="/fs/nexus-projects/sra_search/dnabert_embeds_small_sra_id_map.parquet" \
    --output="dnabert_k_5_small.parquet" \
    --topk=5 \
    --strategy="standard" \

python search.py dnabert \
    --dataset_path="../../../rawbert_data/small.test.jsonl" \
    --accessions_path="../../../rawbert_data/small.accessions.csv" \
    --embeddings_bin_path="/fs/nexus-projects/sra_search/dnabert_embeds_small.bin" \
    --embeddings_idx_map_path="/fs/nexus-projects/sra_search/dnabert_embeds_small_sra_id_map.parquet" \
    --output="dnabert_k_1_small_by_accession.parquet" \
    --topk=1 \
    --strategy="by-accession" \