#!/usr/bin/env bash


SPLITS_DIR=data/splits/
NUM_CLASSES=29
BATCH_SIZE=32
EPOCHS=10
LR=0.001

python3 src/train.py -d $SPLITS_DIR -bs $BATCH_SIZE -c $NUM_CLASSES -e $EPOCHS -lr $LR