# recomm-bert4rec-pytorch

## Dataset
* [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
```bash
cd {project Dir}/datasets/movielens
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

## data preprocess
```shell
python preprocess.py -d 1M
```

## train model
```shell
# [paper]
python train_bert.py -d 1M -v 0.0.0 -k 10 -lr 1e-4 -bs 256 -mp 0.2 -sl 200 -nh 2 -nph 32 -id 256 -nl 3 -op AdamW -e 100

# [best]
python train_bert.py -d 1M -v 0.0.0 -k 10 -lr 1e-3 -bs 64 -mp 0.2 -sl 200 -nh 2 -nph 32 -id 512 -nl 3 -op AdamW -e 200
```
