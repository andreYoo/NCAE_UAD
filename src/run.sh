#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python main.py fmnist fmnist_LeNet ../log/fmnist_test ../data --ratio_pollution 0.2 --lr 0.001 --gan_lr 0.001 --n_epochs 150 --batch_size 128 --weight_decay 0.5e-6 --normal_class 7 --seed 2;
CUDA_VISIBLE_DEVICES=0 python main.py mnist mnist_LeNet ../log/mnist_test ../data --ratio_pollution 0.2 --lr 0.001 --gan_lr 0.001 --n_epochs 150 --batch_size 128 --weight_decay 0.5e-6 --normal_class 0 --seed 2;




