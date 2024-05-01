from shutil import *
import glob, os

# if not os.path.exists("./dataset/train"):
#     os.makedirs("./dataset/train")
# if not os.path.exists("./dataset/val"):
#     os.makedirs("./dataset/val") 
for i in range(18):
    if not os.path.exists(f"./dataset/test_washing_machine_{i}"):
        os.makedirs(f"./dataset/test_washing_machine_{i}")
    # target_train = "./dataset/train"
    # target_val = "./dataset/val"
    target_test = f"./dataset/test_washing_machine_{i}"
 
    original_path = "./dataset/test"
    all_paths = sorted(glob.glob(original_path+'/*'))

    for path_item in all_paths:

        # target_path_item_train = target_train + "/" + path_item.split("/")[-1]
        # target_path_item_val = target_val + "/" + path_item.split("/")[-1]
        target_path_item_test = target_test + "/" + path_item.split("/")[-1]

        # if not os.path.exists(target_path_item_train):
        #     os.makedirs(target_path_item_train)
        # if not os.path.exists(target_path_item_val):
        #     os.makedirs(target_path_item_val)
        if not os.path.exists(target_path_item_test):
            os.makedirs(target_path_item_test)

        if path_item.split("/")[-1] == "washing_machine":
            items = [path_item + '/img_%d.png'%j for j in range(540)]
                # sorted(glob.glob(path_item + '/*.png'))
            for index, item in enumerate(items):
                type = int(index / 30)
                if type == i:
                    copyfile(item, target_path_item_test + '/' + item.split('/')[-1])

