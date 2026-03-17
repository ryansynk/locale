#!/bin/bash

mkdir -p ../indexes/metagraph

CONTIG_DIR=/pscratch/sd/r/rsynk/data/test/logan_contig
OUTPUT_DIR=/pscratch/sd/r/rsynk/rawbert/experiments/sra_recall/indexes/metagraph

# 1. Build the joint canonical graph from your pre-cleaned .fa contigs
find ${CONTIG_DIR} -name "*.fa" | shifter metagraph build -v -k 31 --inplace --mode canonical -p 34 -o ${OUTPUT_DIR}/graph

# 2. Extract primary contigs (Required to prevent 50% sparsity in RowDiff annotations)
shifter metagraph transform -v --to-fasta --primary-kmers -p 34 -o ${OUTPUT_DIR}/primary_contigs ${OUTPUT_DIR}/graph.dbg

# 3. Build the primary graph from the extracted contigs
shifter metagraph build -v -k 31 --mode primary -p 34 -o ${OUTPUT_DIR}/graph_primary ${OUTPUT_DIR}/primary_contigs.fasta.gz

# 4. Create annotation columns for each file separately (Highly efficient since inputs are contigs)
mkdir -p ${OUTPUT_DIR}/columns
find ${CONTIG_DIR} -name "*.fa" | shifter metagraph annotate -v -i ${OUTPUT_DIR}/graph_primary.dbg --anno-filename --separately -o ${OUTPUT_DIR}/columns -p 5 --threads-each 8

# 5. Execute the 3-stage RowDiff transformation on the columns
mkdir -p ${OUTPUT_DIR}/rd_columns
find ${OUTPUT_DIR}/columns -name "*.annodbg" | shifter metagraph transform_anno -v --anno-type row_diff --row-diff-stage 0 -i ${OUTPUT_DIR}/graph_primary.dbg -o ${OUTPUT_DIR}/rd_columns/out -p 34
find ${OUTPUT_DIR}/columns -name "*.annodbg" | shifter metagraph transform_anno -v --anno-type row_diff --row-diff-stage 1 -i ${OUTPUT_DIR}/graph_primary.dbg -o ${OUTPUT_DIR}/rd_columns/out -p 34
find ${OUTPUT_DIR}/columns -name "*.annodbg" | shifter metagraph transform_anno -v --anno-type row_diff --row-diff-stage 2 -i ${OUTPUT_DIR}/graph_primary.dbg -o ${OUTPUT_DIR}/rd_columns/out -p 34

# 6. Transform the delta-coded columns into the Multi-BRWT format
find ${OUTPUT_DIR}/rd_columns/ -name "*.annodbg" | shifter metagraph transform_anno -v --anno-type row_diff_brwt --subsample 10000000 --greedy -i ${OUTPUT_DIR}/graph_primary.dbg -o ${OUTPUT_DIR}/annotation -p 34

# 7. Relax the internal BRWT tree nodes to improve compression
shifter metagraph relax_brwt -v -p 34 --relax-arity 32 -o ${OUTPUT_DIR}/annotation.relaxed ${OUTPUT_DIR}/annotation.row_diff_brwt.annodbg

# Graph is in graph_primary.dbg, annotation is in annotation.relaxed.row_diff_brwt.annodbg