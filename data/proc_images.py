#!/usr/bin/env python
import csv
from PIL import Image
import pickle
import os

img_size = 84

path_data = "/Users/theophilebeaulieu/Desktop/Clement/master_thesis/project/data/"

test_csv_file = path_data + "mini-imagenet_split/test.csv"
train_csv_file = path_data + "mini-imagenet_split/train.csv"
val_csv_file = path_data + "mini-imagenet_split/val.csv"

test_pkl = path_data + "imagenet/mini-imagenet-cache-test.pkl"
train_pkl = path_data + "imagenet/mini-imagenet-cache-train.pkl"
val_pkl = path_data + "imagenet/mini-imagenet-cache-val.pkl"

def pkl_to_raw(csv_file, pkl_file, dirname):
    dir = os.path.join(path_data, dirname)
    if not os.path.isdir(dir):
        os.mkdir(dir)
        print("made: ", dirname)
    p = pickle.load(open(pkl_file, 'rb'))
    imgs = p["image_data"]
    csv_r = csv.reader(open(csv_file))
    i = -1
    last_label = ""
    for (image_filename, class_name) in csv_r:
        # skip headers
        if i == -1:
            i += 1
            continue

        im = Image.fromarray(imgs[i])
        # resize as in maml source code
        im = im.resize((img_size, img_size), resample=Image.LANCZOS)
    
        # make a new subdir for every label
        if class_name != last_label:
            os.mkdir(dir + "/" + class_name)
            last_label = class_name
    
        # save image file into correct class folder
        new_filename = dir + "/" + class_name + "/" +  image_filename
        im.save(new_filename)
        i += 1
        if i % 500 == 0:
            print(i)

pkl_to_raw(train_csv_file, train_pkl, "train")
pkl_to_raw(val_csv_file, val_pkl, "val")
pkl_to_raw(test_csv_file, test_pkl, "test")