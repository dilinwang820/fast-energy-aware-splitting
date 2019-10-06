#! /bin/bash


for((id=1;id<=3;id++))
do
    python summary.py  --data /data/imagenet/  --workers 8 --load saved_models/splitting_${id}.pth.tar
done



