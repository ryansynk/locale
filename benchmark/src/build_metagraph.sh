#!/bin/bash

# Build one metagraph index (primary graph + relaxed row_diff_brwt annotation)
# from a manifest of contig files. Called by MetagraphIndex.build, once per
# node in a multi-node run (each node builds its own shard).
#
# Exit immediately if a command fails (-e), if an unbound variable is used (-u),
# or if a command in a pipeline fails (-o pipefail)
set -euo pipefail

# Assign inputs and die immediately with a specific error if any are missing
METAGRAPH_EXEC=${1:?"Error: Missing metagraph executable (Argument 1)"}
K=${2:?"Error: Missing k value (Argument 2)"}
NUM_THREADS=${3:?"Error: Missing number of threads (Argument 3)"}
CONTIG_MANIFEST=${4:?"Error: Missing contig manifest file path (Argument 4)"}
OUTPUT_DIR=${5:?"Error: Missing output directory path (Argument 5)"}
# RAM available to this build in GB (the node's, or the SLURM per-node limit).
MEM_GB=${6:-400}

# Validate that the manifest actually exists
if [ ! -f "${CONTIG_MANIFEST}" ]; then
    echo "Error: Manifest file does not exist at ${CONTIG_MANIFEST}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Memory caps. Graph construction sorts k-mers in buffers of --mem-cap-gb and,
# with --disk-swap, spills the sorted chunks to disk instead of holding every
# chunk in RAM: the sra500 canonical build (10.9B k-mers) peaked at 132 GB RSS
# without it, and sra4571 has 13x the input. The docs recommend 50-80 GB
# buffers for very large graphs and call anything larger counterproductive.
# transform_anno's --mem-cap-gb defaults to 1000 GB, above any node here.
BUILD_MEM_CAP_GB=$(( MEM_GB / 5 ))
if [ "${BUILD_MEM_CAP_GB}" -gt 80 ]; then BUILD_MEM_CAP_GB=80; fi
if [ "${BUILD_MEM_CAP_GB}" -lt 1 ]; then BUILD_MEM_CAP_GB=1; fi
TRANSFORM_MEM_GB=$(( MEM_GB * 8 / 10 ))
if [ "${TRANSFORM_MEM_GB}" -lt 1 ]; then TRANSFORM_MEM_GB=1; fi
SWAP_DIR="${OUTPUT_DIR}/tmp"
mkdir -p "${SWAP_DIR}"

# In-place construction serializes the succinct graph without first loading it
# into RAM. metagraph 0.5.0 (the ghcr.io image pulled to Perlmutter 2026-03-03)
# exposes it as opt-in --inplace; later upstream builds make it the default and
# only offer the opt-out --in-ram, on which --inplace is an "Unknown option"
# error. Probe the help text so both work.
INPLACE_FLAG=""
if ${METAGRAPH_EXEC} build --advanced --help 2>&1 | grep -q -- '--inplace'; then
    INPLACE_FLAG="--inplace"
fi
echo "metagraph build: mem-cap ${BUILD_MEM_CAP_GB} GB, disk swap ${SWAP_DIR}, ${INPLACE_FLAG:-in-place by default}; transform mem-cap ${TRANSFORM_MEM_GB} GB"

# 1. Build the joint canonical graph from your pre-cleaned .fa contigs
cat "${CONTIG_MANIFEST}" | ${METAGRAPH_EXEC} build -v -k "${K}" --mode canonical -p "${NUM_THREADS}" \
    --mem-cap-gb "${BUILD_MEM_CAP_GB}" --disk-swap "${SWAP_DIR}" ${INPLACE_FLAG} \
    -o "${OUTPUT_DIR}/graph"

# 2. Extract primary contigs (Required to prevent 50% sparsity in RowDiff annotations)
${METAGRAPH_EXEC} transform -v --to-fasta --primary-kmers -p "${NUM_THREADS}" -o "${OUTPUT_DIR}/primary_contigs" "${OUTPUT_DIR}/graph.dbg"

# 3. Build the primary graph from the extracted contigs
${METAGRAPH_EXEC} build -v -k "${K}" --mode primary -p "${NUM_THREADS}" \
    --mem-cap-gb "${BUILD_MEM_CAP_GB}" --disk-swap "${SWAP_DIR}" ${INPLACE_FLAG} \
    -o "${OUTPUT_DIR}/graph_primary" "${OUTPUT_DIR}/primary_contigs.fasta.gz"

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
for stage in 0 1 2; do
    find "${OUTPUT_DIR}/columns" -name "*.annodbg" | ${METAGRAPH_EXEC} transform_anno -v --anno-type row_diff --row-diff-stage "${stage}" \
        --mem-cap-gb "${TRANSFORM_MEM_GB}" -i "${OUTPUT_DIR}/graph_primary.dbg" -o "${OUTPUT_DIR}/rd_columns/out" -p "${NUM_THREADS}"
done

# 6. Transform the delta-coded columns into the Multi-BRWT format
find "${OUTPUT_DIR}/rd_columns/" -name "*.annodbg" | ${METAGRAPH_EXEC} transform_anno -v --anno-type row_diff_brwt --subsample 10000000 --greedy \
    --mem-cap-gb "${TRANSFORM_MEM_GB}" -i "${OUTPUT_DIR}/graph_primary.dbg" -o "${OUTPUT_DIR}/annotation" -p "${NUM_THREADS}"

# 7. Relax the internal BRWT tree nodes to improve compression
${METAGRAPH_EXEC} relax_brwt -v -p "${NUM_THREADS}" --relax-arity 32 -o "${OUTPUT_DIR}/annotation.relaxed" "${OUTPUT_DIR}/annotation.row_diff_brwt.annodbg"

# 8. Drop construction-only files. What the query server reads stays:
# graph_primary.dbg with its .anchors and .rd_succ sidecars, and
# annotation.relaxed.row_diff_brwt.annodbg. The rest (canonical graph,
# primary contigs, per-file columns, row_diff columns, the .pred/.succ
# traversal tables at ~8 bytes per k-mer, the unrelaxed BRWT, swap) is several
# times the index. KEEP_INTERMEDIATES=1 keeps it for debugging.
if [ "${KEEP_INTERMEDIATES:-0}" != "1" ]; then
    rm -rf "${SWAP_DIR}" "${OUTPUT_DIR}/columns" "${OUTPUT_DIR}/rd_columns"
    rm -f "${OUTPUT_DIR}/graph.dbg" "${OUTPUT_DIR}/primary_contigs.fasta.gz" \
          "${OUTPUT_DIR}/annotation.row_diff_brwt.annodbg" \
          "${OUTPUT_DIR}/graph_primary.dbg.pred" "${OUTPUT_DIR}/graph_primary.dbg.pred_boundary" \
          "${OUTPUT_DIR}/graph_primary.dbg.succ" "${OUTPUT_DIR}/graph_primary.dbg.succ_boundary"
fi

# Graph is in graph_primary.dbg, annotation is in annotation.relaxed.row_diff_brwt.annodbg
