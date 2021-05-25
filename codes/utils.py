import cv2
import pandas as pd
import numpy as np
import os

def load_data():
    """
    Loads the train, validation, and test images in 3 separate lists along with 
    their respective labels.

    Results are returned in the form of 3 tuples:
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    """
    # get directory of current file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    # switch working directory to data directory 
    main_dir = os.path.dirname(file_dir)
    os.chdir(main_dir + "\\data")
    # import labels-files.csv as reference
    df = pd.read_csv("labels-files.csv")
    # initialize image and label lists, along with their start indices
    X_train, y_train, idx_train = ([None]*65000, np.empty(shape=(65000,), dtype=int), 0)
    X_valid, y_valid, idx_valid = ([None]*12500, np.empty(shape=(12500,), dtype=int), 0)
    X_test, y_test, idx_test = ([None]*12500, np.empty(shape=(12500,), dtype=int), 0)
    for i in range(90000):
        if df.loc[i,"split"] == "train":
            X_train[idx_train] = cv2.imread(df.loc[i,"path"])
            y_train[idx_train] = df.loc[i,"label_idx"]
            idx_train += 1
        elif df.loc[i,"split"] == "valid":
            X_valid[idx_valid] = cv2.imread(df.loc[i,"path"])
            y_valid[idx_valid] = df.loc[i,"label_idx"]
            idx_valid += 1
        else:       # test data
            X_test[idx_test] = cv2.imread(df.loc[i,"path"])
            y_test[idx_test] = df.loc[i,"label_idx"]
            idx_test += 1
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def get_channel(X, channel):
    """
    Extracts the specific channel of each image in a given list.
    
    Inputs:
    - X: list of images
    - channel: channel to extract ("R", "G", or "B")
    """
    channel_map = {"R": 2, "G": 1, "B": 0}
    Xt = [None]*len(X)
    for (i, img) in enumerate(X):
        assert img.ndim == 3, "Image does not have 3 channels"
        Xt[i] = img[:,:,channel_map[channel]]
    return Xt