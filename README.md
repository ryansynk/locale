# Rawbert

To create an environment for training on nexus, run the following:
```
bash environment.sh
```

To create the dataset, run the following:
```
bash scripts/download_gencode.sh
python scripts/split_gencode.py
```

To train the model, run with the following (works on A6000):
```
python train.py \
--dataset_path="./data/gencode.v49.transcripts.train.fa" \
--test_dataset_path="./data/gencode.v49.transcripts.test.fa" \
--batch_size=8 \
--lr=1e-6 \
--epochs=2 \
--dim=128 \
--moco_queue_size 1024
```

Or just run `bash train.sh`