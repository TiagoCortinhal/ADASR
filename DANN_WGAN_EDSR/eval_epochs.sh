#!/bin/sh

read -p "Folder with the checkpoints: " folder
read -p "Start epoch to eval: " n1
read -p "End epoch to eval: " n2
read -p "Target source? " n3
for i in $(seq $n1 $n2); do ~/anaconda3/bin/python3 main.py --output-dir $folder --eval-epoch $i --factor 8 --eval --train-type $n3 ; done
