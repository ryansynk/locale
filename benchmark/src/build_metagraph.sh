#!/bin/bash

# Exit immediately if a command fails (-e), if an unbound variable is used (-u), 
# or if a command in a pipeline fails (-o pipefail)
set -euo pipefail

# Assign inputs and die immediately with a specific error if any are missing
METAGRAPH_EXEC=${1:?"Error: Missing metagraph executable (Argument 1)"}
K=${2:?"Error: Missing k value (Argument 2)"}
NUM_THREADS=${3:?"Error: Missing number of threads (Argument 3)"}
CONTIG_MANIFEST=${4:?"Error: Missing contig manifest file path (Argument 4)"}
OUTPUT_DIR=${5:?"Error: Missing output directory path (Argument 5)"}

# Validate that the manifest actually exists
if [ ! -f "${CONTIG_MANIFEST}" ]; then
    echo "Error: Manifest file does not exist at ${CONTIG_MANIFEST}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# 1. Build the joint canonical graph from your pre-cleaned .fa contigs
# No --inplace: upstream made in-place construction the default and replaced the
# flag with an opt-out --in-ram. Passing --inplace is a hard "Unknown option"
# error on current metagraph, so this is not backwards compatible with builds
# older than that change (e.g. the Mar 2026 source checkout in ~/metagraph).
cat "${CONTIG_MANIFEST}" | ${METAGRAPH_EXEC} build -v -k "${K}" --mode canonical -p "${NUM_THREADS}" -o "${OUTPUT_DIR}/graph"

# 2. Extract primary contigs (Required to prevent 50% sparsity in RowDiff annotations)
${METAGRAPH_EXEC} transform -v --to-fasta --primary-kmers -p "${NUM_THREADS}" -o "${OUTPUT_DIR}/primary_contigs" "${OUTPUT_DIR}/graph.dbg"

# 3. Build the primary graph from the extracted contigs
${METAGRAPH_EXEC} build -v -k "${K}" --mode primary -p "${NUM_THREADS}" -o "${OUTPUT_DIR}/graph_primary" "${OUTPUT_DIR}/primary_contigs.fasta.gz"

# 4. Create annotation columns for each file separately
mkdir -p "${OUTPUT_DIR}/columns"

# Calculate optimal parallel jobs for annotation to prevent CPU oversubscription
THREADS_EACH=8
ANNO_P=$(( NUM_THREADS / THREADS_EACH ))
if [ "$ANNO_P" -lt 1 ]; then 
    ANNO_P=1
    THREADS_EACH=$NUM_THREADS
fi

cat "${CONTIG_MANIFEST}" | ${METAGRAPH_EXEC} annotate -v -i "${OUTPUT_DIR}/graph_primary.dbg" --anno-filename --separately -o "${OUTPUT_DIR}/columns" -p "${ANNO_P}" --threads-each "${THREADS_EACH}"

# 5. Execute the 3-stage RowDiff transformation on the columns
mkdir -p "${OUTPUT_DIR}/rd_columns"
find "${OUTPUT_DIR}/columns" -name "*.annodbg" | ${METAGRAPH_EXEC} transform_anno -v --anno-type row_diff --row-diff-stage 0 -i "${OUTPUT_DIR}/graph_primary.dbg" -o "${OUTPUT_DIR}/rd_columns/out" -p "${NUM_THREADS}"
find "${OUTPUT_DIR}/columns" -name "*.annodbg" | ${METAGRAPH_EXEC} transform_anno -v --anno-type row_diff --row-diff-stage 1 -i "${OUTPUT_DIR}/graph_primary.dbg" -o "${OUTPUT_DIR}/rd_columns/out" -p "${NUM_THREADS}"
find "${OUTPUT_DIR}/columns" -name "*.annodbg" | ${METAGRAPH_EXEC} transform_anno -v --anno-type row_diff --row-diff-stage 2 -i "${OUTPUT_DIR}/graph_primary.dbg" -o "${OUTPUT_DIR}/rd_columns/out" -p "${NUM_THREADS}"

# 6. Transform the delta-coded columns into the Multi-BRWT format
find "${OUTPUT_DIR}/rd_columns/" -name "*.annodbg" | ${METAGRAPH_EXEC} transform_anno -v --anno-type row_diff_brwt --subsample 10000000 --greedy -i "${OUTPUT_DIR}/graph_primary.dbg" -o "${OUTPUT_DIR}/annotation" -p "${NUM_THREADS}"

# 7. Relax the internal BRWT tree nodes to improve compression
${METAGRAPH_EXEC} relax_brwt -v -p "${NUM_THREADS}" --relax-arity 32 -o "${OUTPUT_DIR}/annotation.relaxed" "${OUTPUT_DIR}/annotation.row_diff_brwt.annodbg"

# Graph is in graph_primary.dbg, annotation is in annotation.relaxed.row_diff_brwt.annodbg