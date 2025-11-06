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

# Get cuttlefish
git clone git@github.com:COMBINE-lab/cuttlefish.git
cd cuttlefish/
mkdir build && cd build/
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make -j 8 install
cd ../..
ulimit -n 2048
cd ${SCRIPT_DIR}

python ${SCRIPT_DIR}/make_dataset.py ${SCRIPT_DIR}/../data/dataset/gencode.v49.transcripts.fa