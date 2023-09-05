#!/usr/bin/env python3

import argparse
import numpy as np
from typing import List
from pathlib import Path
import cv2
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


@dataclass
class Image:
    data:np.ndarray
    label:str

def get_data(data_dir:str) -> List[Image]:
    images = []
    for label_dir in Path(data_dir).iterdir():
        for image in label_dir.iterdir():
            img = cv2.imread(image.as_posix())
            img = cv2.resize(img, (32, 32))
            images.append(Image(img, label_dir.name))
    return images


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_path",
                        default="data/raw/asl_alphabet_train/asl_alphabet_train/")
    
    parser.add_argument("-o",
                        "--output_path",
                        default="data/splits/")
    
    args = parser.parse_args()
    encoder = OneHotEncoder(sparse_output=False)
    
    all_images = get_data(args.input_path)
    
    images = np.array([img.data for img in all_images])
    images_normalized = images.astype("float32") / 255.0 # normalize each pixel value
    
    labels = [img.label for img in all_images]
    one_hot_labels = encoder.fit_transform(np.array(labels).reshape(-1,1))
    
    X_train, X_test, y_train, y_test = train_test_split(images_normalized, 
                                                        one_hot_labels,
                                                        test_size=.2)

    for array, fname in [(X_train, "X_train.npy"), (X_test, "X_test.npy"), (y_train, "y_train.npy"), (y_test, "y_test.npy")]:
        np.save(f"{args.output_path}/{fname}", array)
    

if __name__ == "__main__":
    main()