#!/bin/bash

#python train.py --dataset WADI --epochs 10 --bs 64
#python train.py --dataset WADI --epochs 5 --bs 64 --init_lr 0.001
python train.py --dataset WADI --epochs 3 --bs 64 --init_lr 0.001 --lookback 80
#python train.py --dataset WADI --epochs 5 --bs 64 --init_lr 1 --lookback 80
#python train.py --dataset WADI --epochs 20 --bs 64
#python train.py --dataset WADI --epochs 30 --bs 64
#python train.py --dataset WADI --epochs 50 --bs 64
#python train.py --dataset WADI --epochs 10 --bs 128