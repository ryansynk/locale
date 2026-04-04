# CLAUDE.md

## Project
Creating an embedding model for local alignment of DNA sequences.
The goal is to convert sequence search over the large sets of sequences (like NIH Sequence Read Archive) into vector search. Vector search could be scalable and importantly robust to noise (an advantage over k-mer methods like metagraph)

## Experiments
Current experiment is in experiments/sra_recall
- Tests performance of searching over a set of accessions. 
- Each methods index.py file contains code for building and searching of accessions and searching with queries. 
- metagraph_index contains code for k-mer search
- dense_index contains code for a general vector embedding model, with different embedding models as subclasses.
- Each method returns a score for (up to) every accession.
- plot_results takes these scores, sorts them, and calculates recall @ k

## Training
- DNABERT base model (could change)
- Momentum Contrast training strategy
- Self-supervised dataset generated from a parent contig into two crops (containment or overlap)
- Noise added to crops
- Configs for training found in /configs

## Commands
srun uv run python -m torch.distributed.run --nnodes=4 --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py --config configs/unsupervised_perlmutter_containment.yaml

uv run python run_benchmark.py --config configs/perlmutter_rawbert.yaml --model.checkpoint_path /pscratch/sd/r/rsynk/rawbert/checkpoints/ge6jbfvp/checkpoint7000.pth.tar --model.pooling mean --model.max_seq_len 256 --model.chunk_type stride --query_type gencode --mutation_rate 0.0

uv run python plot_results.py results/ /pscratch/sd/r/rsynk/rawbert_data/data/sra_recall/raw_read_queries_final.parquet