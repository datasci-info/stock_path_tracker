#!/bin/bash

DIRS=('is_turnpt_upward' 'is_turnpt_downward')
Ns=(1 2 3 5)

for direction in "${DIRS[@]}"
do
    for n in ${Ns[@]}
    do
        export N=$n
        export DIRECTION=$direction
        export LEARNING_RATE=0.1
        python model.py  > models/${N}_${direction}.log 2>&1 &
    done

done

wait

for direction in "${DIRS[@]}"
do
    for n in ${Ns[@]}
    do
        tail -1 models/${n}_${direction}.log >> model_result.csv
    done

done


