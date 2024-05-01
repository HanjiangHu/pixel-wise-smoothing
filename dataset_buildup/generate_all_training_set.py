from shutil import *
import glob, os
import cupy as np
import numpy
import pickle

 
all_trans_paths = sorted(glob.glob('../metaroom/*'))
metaroom_training_all = "../metaroom_all"
for trans in all_trans_paths:
    print(trans)
    trans_train_items = sorted(glob.glob(f'{trans}/train/*'))
    for item in trans_train_items:
        print(item)
        img_paths = sorted(glob.glob(f'{item}/*'))
        metaroom_training_all_item = metaroom_training_all + '/train/' + item.split('/')[-1]
        if not os.path.exists(metaroom_training_all_item):
            os.makedirs(metaroom_training_all_item)
        for img in img_paths: 
            copyfile(img, metaroom_training_all_item + '/' + trans.split('/')[-1] + '_' + img.split('/')[-1])
    trans_val_items = sorted(glob.glob(f'{trans}/val/*'))
    for item in trans_val_items:
        print(item)
        img_paths = sorted(glob.glob(f'{item}/*'))
        metaroom_training_all_item = metaroom_training_all + '/train/' + item.split('/')[-1]
        if not os.path.exists(metaroom_training_all_item):
            os.makedirs(metaroom_training_all_item)
        for img in img_paths:
            copyfile(img, metaroom_training_all_item + '/' + trans.split('/')[-1] + '_' + img.split('/')[-1])

trans_test_items = sorted(glob.glob('../metaroom/metaroom_tz_vanilla/proj_test/*'))
for item in trans_test_items:
    img_paths = sorted(glob.glob(f'{item}/*'))
    metaroom_training_all_item = metaroom_training_all + '/test/' + item.split('/')[-1]
    if not os.path.exists(metaroom_training_all_item):
        os.makedirs(metaroom_training_all_item)
    for img in img_paths:
        copyfile(img, metaroom_training_all_item + '/' + img.split('/')[-1])


