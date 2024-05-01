from shutil import *
import glob, os
import cupy as np
import pickle
# NEW_NAME = "../metaroom/metaroom_ty_va"
NEW_NAME = "./metaroom_partial/metaroom_ry_va"
# OLD_TRAIN = "projected_imgs_tz"
# OLD_TEST = "projected_imgs"
# OLD_TEST_BETA = "projected_imgs_tz"
#
# OLD_TEST_UNIFORM = "projected_imgs_tz"

OLD_CERTIFY = "projected_large_reslt_imgs_ry"
PC_to_be_merged = "projected_imgs"

 
if not os.path.exists(f"../{NEW_NAME}/certify"):
    os.makedirs(f"../{NEW_NAME}/certify")
target_certify = f"../{NEW_NAME}/certify"
certify_path = f"../{OLD_CERTIFY}"
PC_to_be_merged_path = f"../{PC_to_be_merged}"
all_certify_paths = sorted(glob.glob(certify_path+'/*'))
all_one_frame_pc_tobemerged_paths = sorted(glob.glob(PC_to_be_merged_path+'/*'))
for index, path_item in enumerate(all_certify_paths):
    target_path_item_certify = target_certify + "/" + path_item.split("/")[-1]
    if not os.path.exists(target_path_item_certify):
        os.makedirs(target_path_item_certify)
    # print(all_test_paths[index] + '/test/*.png')
    # copyfile(all_certify_paths[index] + '/camera_intrinsic_matrix.npy', target_path_item_certify + '/' + item.split('/')[-1])
    save_pickle = {}
    intrinsic_matrix = np.load(all_certify_paths[index] + '/camera_intrinsic_matrix.npy')
    save_pickle["intrinsic_matrix"] = intrinsic_matrix
    pc_items = sorted(glob.glob(all_certify_paths[index] + '/new_test*.npy'))
    # for item in pc_items:
    #     print(111111111111, item)
    #     copyfile(item, target_path_item_certify + '/' + item.split('/')[-1])
    pose_items = sorted(glob.glob(all_certify_paths[index] + '/test/*.npy'))
    one_pc_to_be_merged = sorted(glob.glob(all_one_frame_pc_tobemerged_paths[index] + '/test*.npy'))
    for item in pose_items:
        print(111111111111, item)
        for ind, pc in enumerate(pc_items):
            if item.split('/')[-1][5:-6] == pc.split('/')[-1][9:-7]:
                save_pickle["pose"] = np.load(item)
                save_pickle["point_cloud"] = np.load(pc)
                print("np.load(pc)", np.load(pc).shape)
                assert pc_items[ind] == pc
                save_pickle["one_frame_point_cloud"] = np.load(one_pc_to_be_merged[ind])
                print("np.load(one_pc_to_be_merged[ind])", np.load(one_pc_to_be_merged[ind]).shape)
                break
        with open(target_path_item_certify + '/' + item.split('/')[-1][:-3]+"pkl", 'wb') as f:
            pickle.dump(save_pickle, f)
        # copyfile(item, )
