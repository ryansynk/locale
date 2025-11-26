#!/usr/bin/env bash
CURR_DIR="$(pwd)"
SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPT_DIR}

# Download and extract gencode human transcriptome
echo "Downloading transcriptome"
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.transcripts.fa.gz
mkdir ${SCRIPT_DIR}/../data
mv ${SCRIPT_DIR}/gencode.v49.transcripts.fa.gz ${SCRIPT_DIR}/../data
cd ${SCRIPT_DIR}/../data
gunzip gencode.v49.transcripts.fa.gz
cd ${CURR_DIR}