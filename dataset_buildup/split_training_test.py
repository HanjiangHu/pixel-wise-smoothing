from shutil import *
import glob, os
import cupy as np
import numpy
import pickle
# NEW_NAME = "../metaroom/metaroom_ty_va"
# NEW_NAME = "../metaroom_1_5/metaroom_tz"
NEW_NAME = "../metaroom/metaroom_tz"
OLD_TRAIN = "1_5_and_projected_imgs_tz" #"projected_imgs_tz"
OLD_TEST = "projected_imgs" 
OLD_TEST_BETA = "projected_imgs_tz"
  
OLD_TEST_UNIFORM = "projected_imgs_tz"

OLD_CERTIFY = "projected_large_reslt_imgs_tz"

OLD_TEST_LANDSCAPE = "projected_imgs_tz"

'''
if not os.path.exists(f"../{NEW_NAME}/certify"):
    os.makedirs(f"../{NEW_NAME}/certify")
target_certify = f"../{NEW_NAME}/certify"
certify_path = f"../../scan_controller/{OLD_CERTIFY}"
all_certify_paths = sorted(glob.glob(certify_path+'/*'))
for index, path_item in enumerate(all_certify_paths):
    target_path_item_certify = target_certify + "/" + path_item.split("/")[-1]
    if not os.path.exists(target_path_item_certify):
        os.makedirs(target_path_item_certify)
    # print(all_test_paths[index] + '/test/*.png')
    # copyfile(all_certify_paths[index] + '/camera_intrinsic_matrix.npy', target_path_item_certify + '/' + item.split('/')[-1])
    save_pickle = {}
    intrinsic_matrix = np.load(all_certify_paths[index] + '/camera_intrinsic_matrix.npy')
    save_pickle["intrinsic_matrix"] = intrinsic_matrix
    pc_items = sorted(glob.glob(all_certify_paths[index] + '/test*.npy'))
    # for item in pc_items:
    #     print(111111111111, item)
    #     copyfile(item, target_path_item_certify + '/' + item.split('/')[-1])
    pose_items = sorted(glob.glob(all_certify_paths[index] + '/test/*.npy'))
    for item in pose_items:
        print(111111111111, item)
        for pc in pc_items:
            if item.split('/')[-1][5:-6] == pc.split('/')[-1][5:-7]:
                save_pickle["pose"] = np.load(item)
                save_pickle["point_cloud"] = np.load(pc)
                break
        with open(target_path_item_certify + '/' + item.split('/')[-1][:-3]+"pkl", 'wb') as f:
            pickle.dump(save_pickle, f)
        # copyfile(item, )
'''

if not os.path.exists(f"../{NEW_NAME}/train"): 
    os.makedirs(f"../{NEW_NAME}/train")
if not os.path.exists(f"../{NEW_NAME}/val"):
    os.makedirs(f"../{NEW_NAME}/val")
if not os.path.exists(f"../{NEW_NAME}/test"):
    os.makedirs(f"../{NEW_NAME}/test")
if not os.path.exists(f"../{NEW_NAME}/test_beta"):
    os.makedirs(f"../{NEW_NAME}/test_beta")
if not os.path.exists(f"../{NEW_NAME}/test_uniform"):
    os.makedirs(f"../{NEW_NAME}/test_uniform")
if not os.path.exists(f"../{NEW_NAME}/test_landscape"):
    os.makedirs(f"../{NEW_NAME}/test_landscape")
target_train = f"../{NEW_NAME}/train"
target_val = f"../{NEW_NAME}/val"
target_test = f"../{NEW_NAME}/test"
target_test_beta = f"../{NEW_NAME}/test_beta"
target_test_uniform = f"../{NEW_NAME}/test_uniform"
target_test_landscape = f"../{NEW_NAME}/test_landscape"

original_path = f"../../scan_controller/{OLD_TRAIN}"
all_paths = sorted(glob.glob(original_path+'/*'))
test_path = f"../../scan_controller/{OLD_TEST}"
all_test_paths = sorted(glob.glob(test_path+'/*'))

test_path_beta = f"../../scan_controller/{OLD_TEST_BETA}"
all_test_beta_paths = sorted(glob.glob(test_path_beta+'/*'))

test_path_uniform = f"../../scan_controller/{OLD_TEST_UNIFORM}"
all_test_uniform_paths = sorted(glob.glob(test_path_uniform+'/*'))

test_path_landscape = f"../../scan_controller/{OLD_TEST_LANDSCAPE}"
all_test_path_landscape_paths = sorted(glob.glob(test_path_landscape+'/*'))




# tz tx ty rz rx ry
'''
for index, path_item in enumerate(all_paths):
    items = sorted(glob.glob(path_item + '/train/*.png'))
    print(path_item)
    assert path_item == all_paths[index]

    target_path_item_train = target_train + "/" + path_item.split("/")[-1]
    target_path_item_val = target_val + "/" + path_item.split("/")[-1]


    if not os.path.exists(target_path_item_train):
        os.makedirs(target_path_item_train)
    if not os.path.exists(target_path_item_val):
        os.makedirs(target_path_item_val)

    random_val = numpy.random.randint(0, len(items)-1, size=int(0.1*len(items)))
    print(len(random_val))
    for index, item in enumerate(items):
        if index in random_val:
            copyfile(item, target_path_item_val+'/'+item.split('/')[-1])
            # if write_txt_flag:
            #     with open("ImageSets/test.txt","a") as f:
            #         f.write(item.split('/')[-1].split('.')[-2])
            #         f.write('\n')
            #     with open("ImageSets/val.txt","a") as f:
            #         f.write(item.split('/')[-1].split('.')[-2])
            #         f.write('\n')
        else:
            copyfile(item, target_path_item_train + '/' + item.split('/')[-1])
            # if write_txt_flag:
                # with open("ImageSets/train.txt","a") as f:
                #     f.write(item.split('/')[-1].split('.')[-2])
                #     f.write('\n')
    # print(target_path_item_train, target_path_item_val, target_path_item_test)
'''
'''
# va
for index, path_item in enumerate(all_paths):
    items = sorted(glob.glob(path_item + '/train/*.png'))
    print(path_item)
    assert path_item == all_paths[index]

    target_path_item_train = target_train + "/" + path_item.split("/")[-1]
    target_path_item_val = target_val + "/" + path_item.split("/")[-1]


    if not os.path.exists(target_path_item_train):
        os.makedirs(target_path_item_train)
    if not os.path.exists(target_path_item_val):
        os.makedirs(target_path_item_val)

    # random_val = numpy.random.randint(0, len(items)-1, size=int(0.1*len(items)))
    # print(len(random_val))
    for index, item in enumerate(items):
        if index % 20 != 0: continue
        if numpy.random.rand() < 0.02:
            copyfile(item, target_path_item_val+'/'+item.split('/')[-1])
            # if write_txt_flag:
            #     with open("ImageSets/test.txt","a") as f:
            #         f.write(item.split('/')[-1].split('.')[-2])
            #         f.write('\n')
            #     with open("ImageSets/val.txt","a") as f:
            #         f.write(item.split('/')[-1].split('.')[-2])
            #         f.write('\n')
        else:
            copyfile(item, target_path_item_train + '/' + item.split('/')[-1])
            # if write_txt_flag:
                # with open("ImageSets/train.txt","a") as f:
                #     f.write(item.split('/')[-1].split('.')[-2])
                #     f.write('\n')
    # print(target_path_item_train, target_path_item_val, target_path_item_test)
'''

'''
for index, path_item in enumerate(all_test_paths):
    target_path_item_test = target_test + "/" + path_item.split("/")[-1]
    # print(all_test_paths[index] + '/test/*.png')
    test_items = sorted(glob.glob(all_test_paths[index] + '/test/*.png'))
    if not os.path.exists(target_path_item_test):
        os.makedirs(target_path_item_test)
    for item in test_items:
        # print(111111111111, item, target_path_item_test)
        copyfile(item, target_path_item_test + '/' + item.split('/')[-1])

for index, path_item in enumerate(all_test_beta_paths):
    target_path_item_test = target_test_beta + "/" + path_item.split("/")[-1]
    # print(all_test_paths[index] + '/test/*.png')
    test_items = sorted(glob.glob(all_test_beta_paths[index] + '/test_beta/*.png'))
    if not os.path.exists(target_path_item_test):
        os.makedirs(target_path_item_test)
    for item in test_items:
        # print(111111111111, item, target_path_item_test)
        copyfile(item, target_path_item_test + '/' + item.split('/')[-1])

for index, path_item in enumerate(all_test_uniform_paths):
    target_path_item_test = target_test_uniform + "/" + path_item.split("/")[-1]
    # print(all_test_paths[index] + '/test/*.png')
    test_items = sorted(glob.glob(all_test_uniform_paths[index] + '/test_uniform/*.png'))
    if not os.path.exists(target_path_item_test):
        os.makedirs(target_path_item_test)
    for item in test_items:
        # print(111111111111, item, target_path_item_test)
        copyfile(item, target_path_item_test + '/' + item.split('/')[-1])
'''
for index, path_item in enumerate(all_test_path_landscape_paths):
    target_path_item_test = target_test_landscape + "/" + path_item.split("/")[-1]
    # print(all_test_paths[index] + '/test/*.png')
    test_items = sorted(glob.glob(all_test_path_landscape_paths[index] + '/test_landscape/*.png'))
    if not os.path.exists(target_path_item_test):
        os.makedirs(target_path_item_test)
    for item in test_items:
        # print(111111111111, item, target_path_item_test)
        r_name = item.split("/")[-1][5:-4].split("_")[0]
        y_name = item.split("/")[-1][5:-4].split("_")[1][1:]
        num_name = item.split("/")[-1][5:-4].split("_")[2]
        together = r_name + "_" + y_name + "_" + num_name
        if together in ["28_59_4", "28_119_5", "28_-60_2", "28_-120_1"]:
            copyfile(item, target_path_item_test + '/' + item.split('/')[-1])




