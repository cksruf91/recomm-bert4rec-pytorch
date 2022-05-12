# recomm-bert4rec-pytorch

## Dataset
* [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
```bash
cd {project Dir}/datasets/movielens
cd datasets/movielens
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

## data preprocess
```shell
python preprocess.py -d 1M
```

## train model
```shell
python train_bert.py -d 1M -v 0.0.0 -k 10
```
