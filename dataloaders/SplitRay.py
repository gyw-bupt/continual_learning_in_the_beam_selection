import numpy as np
import torch
from random import shuffle

from .Raymobtime import CustomDataset

def SplitRay():

    train_files = ["./data_train_valid_test/los_train.npz",
                   "./data_train_valid_test/nlos_train.npz"]

    val_files = ["./data_train_valid_test/los_validation.npz",
                  "./data_train_valid_test/nlos_validation.npz"]

    test_files = ["./data_train_valid_test/los_test.npz",
                 "./data_train_valid_test/nlos_test.npz"]

    train_dataset_splits = {}
    test_dataset_splits = {}
    val_dataset_splits = {}

    for index, train_x_file in enumerate(train_files, start=1):
        train_dataset = CustomDataset(train_x_file, transform=None)
        train_dataset_splits[index] = train_dataset

    for index, test_x_file in enumerate(test_files, start=1):
        test_dataset = CustomDataset(test_x_file, transform=None)
        test_dataset_splits[index] = test_dataset

    for index, val_x_file in enumerate(val_files, start=1):
        val_dataset = CustomDataset(val_x_file, transform=None)
        val_dataset_splits[index] = val_dataset

    task_output_space = [256, 256]

    return train_dataset_splits, val_dataset_splits,test_dataset_splits, task_output_space
