#!/usr/bin/env bash
# Experiment on MNIST -> USPS (10 runs)

for SEED in 1 2 3 4 5 6 7 8 9 10
do
  for SAMPLES in 1 3 5 7
  do
    python main_sda.py \
        --cfg cfg/digits-s-1.json \
        --src MT \
        --tgt US \
        --method dsne \
        --nc 10 \
        --size 32 \
        --bb baseline2convs \
        --dropout \
        --train-src \
        --lr 0.1 \
        --gpus 0 \
        --end-epoch 100 \
        --lr-epoch 50 \
        --log-itv 0 \
        --postfix s$SAMPLES-0.5-l2n-seed$SEED \
        --embed-size 84 \
        --l2n \
        --alpha 0.5 \
        --seed $SEED
  done
done