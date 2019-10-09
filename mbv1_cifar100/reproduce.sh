#! /bin/bash


python summary.py --dataset cifar100 --load ./saved_models/splitting_0.pth.tar --model-name split_finetune

for((id=1;id<8;id++))
do
    python summary.py --dataset cifar100 --load ./saved_models/splitting_${id}.pth.tar --model-name split_finetune --sp
done

