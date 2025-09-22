#!/usr/bin/env bash
SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPT_DIR}

# Download and extract gencode human transcriptome
echo "Downloading transcriptome"
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.transcripts.fa.gz
mkdir ${SCRIPT_DIR}/../data
mkdir ${SCRIPT_DIR}/../data/dataset
mv ${SCRIPT_DIR}/gencode.v49.transcripts.fa.gz ${SCRIPT_DIR}/../data/dataset
cd ${SCRIPT_DIR}/../data/dataset
gunzip gencode.v49.transcripts.fa.gz
cd ${SCRIPT_DIR}

# Download and extract ART sequencing simulator
echo "Downloading ART simulator"
mkdir ${SCRIPT_DIR}/../data/tools
cd ${SCRIPT_DIR}/../data/tools
wget https://www.niehs.nih.gov/sites/default/files/2024-02/artbinmountrainier2016.06.05linux64.tgz
tar -xvf artbinmountrainier2016.06.05linux64.tgz
cd ${SCRIPT_DIR}

echo "Selecting random transcriptome subsets"
python ${SCRIPT_DIR}/select_sequences.py ${SCRIPT_DIR}/../data/dataset/gencode.v49.transcripts.fa

echo "Simulating illumina sequencer"
python ${SCRIPT_DIR}/run_art.py

echo "Finalizing dataset"
python ${SCRIPT_DIR}/make_pairs.py --data_dir "${SCRIPT_DIR}/../data"
