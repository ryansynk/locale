# rawbert

To create an environment for training on nexus, run the following:
```
bash environment.sh
```

To create the dataset, run the following:
```
bash scripts/download_gencode.sh
python scripts/split_gencode.py
```
