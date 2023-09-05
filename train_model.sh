#!/usr/bin/env bash


SPLITS_DIR=data/splits/
CLASSES=29
BATCH_SIZE=32
EPOCHS=15
LEARNING_RATE=0.001

python3 src/models/train_model.py -d $SPLITS_DIR