#!/usr/bin/env bash


SPLITS_DIR=data/splits/
NUM_CLASSES=29
BATCH_SIZE=32
EPOCHS=15
LEARNING_RATE=0.001

python3 src/train.py -d $SPLITS_DIR -c $NUM_CLASSES