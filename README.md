# Project 2 - part1
put `train_modify_simpleNN.py` and `train_direct_use_tensorflow.py` with below settings will generate the two .txt in this repository
## Modification of simpleNN
```
python3 train_modify_simpleNN.py \ 
              --optim SGD --lr 0.001 --C 0.01 \
              --net CNN_4layers --bsize 200 \
              --train_set ./simpleNN/Python/data/mnist-demo.mat \
              --dim 28 28 1 --seed 1 --epoch_max 1
```
## Direct use tensorflow
```
python3 train_direct_use_tensorflow.py \
              --optim SGD --lr 0.001 --C 0.01 \
              --net CNN_4layers --bsize 200 \
              --train_set ./simpleNN/Python/data/mnist-demo.mat \
              --dim 28 28 1 --seed 1 --epoch_max 1
```
