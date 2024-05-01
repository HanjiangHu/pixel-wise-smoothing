import os
import sys

sys.path.append('.')

import os
import math

# evaluate a smoothed classifier on a dataset 
import argparse
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import StrictRotationSmooth
from time import time
import torch
import datetime

from architectures import get_architecture
from transformers_ import RotationTransformer, gen_transformer, DiffResolvableProjectionTransformer
from transforms import visualize
import cupy as np
import numpy
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator, LinearNDInterpolator
from cupy import linalg as LA
import open3d as o3d
import pandas as pd
import glob, os
from tqdm import trange

'''
CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_tz data/alias/metaroom/tz/tv_noise_0.1_0.1 --partial 0.1 
CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_ty data/alias/metaroom/ty/tv_noise_0.05_0.1 --partial 0.05 
CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_tx data/alias/metaroom/tx/tv_noise_0.05_0.1 --partial 0.05
CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_rx data/alias/metaroom/rx/tv_noise_2.5_0.1 --partial 0.04363
CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_ry data/alias/metaroom/ry/tv_noise_2.5_0.1 --partial 0.04363
CUDA_VISIBLE_DEVICES=1 python diff_aliasing_analyze.py metaroom diff_resolvable_rz data/alias/metaroom/rz/tv_noise_7_0.1 --partial 0.122173

CUDA_VISIBLE_DEVICES=0 python diff_aliasing_analyze.py metaroom diff_resolvable_rz data/alias_save/metaroom/rz/tv_noise_0.7_9000 --partial 0.0122173 --
save_k_samples 9000 --not_entire --k 100000000
'''

EPS_PIXEL = 1e-2
# RESOLUTION = 10000
# 100: exact 51, L: 924 ?
# RESOLUTION = 1000 # 1000: exact: 501  L: 770330
# RESOLUTION = 10000 # 10000: 5001, L: 612763
parser = argparse.ArgumentParser(description='Strict rotation certify')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument('transtype', type=str, help='type of projective transformations',
                    choices=['resolvable_tz', 'resolvable_tx', 'resolvable_ty', 'resolvable_rz', 'resolvable_rx',
                             'resolvable_ry', 'diff_resolvable_tz', 'diff_resolvable_tx', 'diff_resolvable_ty',
                             'diff_resolvable_rz', 'diff_resolvable_rx',
                             'diff_resolvable_ry'])
parser.add_argument("aliasfile", type=str, help='output of alias data')
parser.add_argument("--start", type=int, default=0, help="start before skipping how many examples")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--split", default="certify", help="train or test set")
parser.add_argument("--slice", type=int, default=1000, help="number of angle slices")
parser.add_argument("--subslice", type=int, default=500, help="number of subslices for maximum l2 estimation")
parser.add_argument("--partial", type=float, default=180.0, help="only contain +-angle maximum aliasing")
parser.add_argument("--verbstep", type=int, default=10, help="print for how many subslices")
parser.add_argument("--L1_threshold", type=float, default=0.2, help="used for check consistency")
parser.add_argument("--k_ambiguity", type=int, default=1000, help="k ambiguity")
parser.add_argument("--k", type=int, default=100000000 , help="k downsampling")
parser.add_argument("--density", type=float, default=-1, help="voxel downsampling")
parser.add_argument("--not_entire", action='store_true')
parser.add_argument("--save_k_samples",  type=int, default=-1, help="k ambiguity")
parser.add_argument("--exact", action='store_true')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--small_img", action='store_true')
parser.add_argument("--resol",  type=int, default=10000, help="k ambiguity")

args = parser.parse_args()
RESOLUTION = args.resol

def down_sampling(point_cloud_npy, density, k=-1, k_first=True):
    if k > point_cloud_npy.shape[0]:
            return None
    if density > 0:
        # point_cloud_o3d = o3d.t.geometry.PointCloud()#.cuda()
        # point_cloud_o3d.point["positions"] = o3d.core.Tensor(point_cloud_npy[:, 0: 3], o3d.core.float32)#.cuda()
        # point_cloud_o3d.point["colors"] = o3d.core.Tensor(point_cloud_npy[:, 3: 6], o3d.core.float32)#.cuda()
        # point_cloud_o3d = point_cloud_o3d.voxel_down_sample(density)
        # original_positions = np.asarray((point_cloud_o3d.cpu().point["positions"]).numpy())
        # colors = np.asarray((point_cloud_o3d.cpu().point["colors"]).numpy())

        # #point_cloud_npy = np.asnumpy(point_cloud_npy)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_npy[:, 0: 3])
        point_cloud.colors = o3d.utility.Vector3dVector(point_cloud_npy[:, 3: 6])
        print("start down sampling")
        print(np.asarray(point_cloud.points).shape)
        if k > 0:
            if k_first:
                point_cloud = o3d.geometry.PointCloud.uniform_down_sample(point_cloud, k)
                point_cloud = o3d.geometry.PointCloud.voxel_down_sample(point_cloud, density)
            else:
                point_cloud = o3d.geometry.PointCloud.voxel_down_sample(point_cloud, density)
                point_cloud = o3d.geometry.PointCloud.uniform_down_sample(point_cloud, k)
        else:
            point_cloud = o3d.geometry.PointCloud.voxel_down_sample(point_cloud, density)
        print(np.asarray(point_cloud.points).shape)
        original_positions = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        return np.hstack([original_positions, colors])
    else:
        return np.asnumpy(point_cloud_npy)


def check_consistency(rgb1, rgb2, threshold):
    if (rgb1[0] == 1. and rgb1[1] == 1. and rgb1[2] == 1.) or (rgb2[0] == 1. and rgb2[1] == 1. and rgb2[2] == 1.):
        return False, True
    # print(LA.norm(rgb2 - rgb1, ord=1))
    return True, LA.norm(rgb2 - rgb1, ord=1) < threshold


def filter_frustum(x_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start,
                   colors):  # , point_cloud=np.array(0)):
    project_positions_flat = project_positions_flat[x_ge_0_index]
    project_positions_float = project_positions_float[x_ge_0_index]
    project_positions = project_positions[x_ge_0_index]
    points_start = points_start[x_ge_0_index]
    colors = colors[x_ge_0_index]
    # if len(point_cloud.shape) != 0:
    #     point_cloud = point_cloud[x_ge_0_index]
    # print(points_start.shape)
    return project_positions_flat, project_positions_float, project_positions, points_start, colors  # , point_cloud


def projection_oracle(point_cloud_npy, extrinsic_matrix, intrinsic_matrix, k_ambiguity, round=False, no_filter_frustum=False):
    # load point cloud
    # point_cloud_npy = np.load(f"dataset_old/{object_name}/point_cloud.npy")

    # point_cloud = o3d.io.read_point_cloud(f"dataset/{object_name}/point_cloud.npy", format='xyzrgb', remove_nan_points=True, remove_infinite_points=True, print_progress=False)
    # print(point_cloud)
    #
    # point_cloud.points = open3d.utility.Vector3dVector(point_cloud)
    # print(point_cloud_npy.shape)
    point_cloud = point_cloud_npy
    original_positions = point_cloud[:, 0: 3].astype(np.float16)
    colors = point_cloud[:, 3: 6].astype(np.float16)
    # point_cloud_float16 = point_cloud.astype(np.float16)
    # print(point_cloud_npy.shape[0], np.asarray(point_cloud.points).shape[0])

    # load intrinsic matrix
    # intrinsic_matrix = np.load(f"dataset/{object_name}/camera_intrinsic_matrix.npy")
    # print(intrinsic_matrix)

    # print(intrinsic_matrix)

    # positions = point_cloud[:, 0: 3]
    # colors = point_cloud[:, 3: 6]
    # point_num = point_cloud.shape[0]

    # point_num = original_positions.shape[0]

    # print(original_positions.shape, np.ones((original_positions.shape[0], 1)).shape)
    positions = np.hstack([original_positions, np.ones((original_positions.shape[0], 1))])
    # print(intrinsic_matrix, extrinsic_matrix)
    points_start = (np.linalg.inv(extrinsic_matrix)[0: 3] @ positions.T).T
    project_positions = intrinsic_matrix @ np.linalg.inv(extrinsic_matrix)[0: 3] @ positions.T
    project_positions = project_positions.T
    project_positions_float = project_positions[:, 0:2] / project_positions[:, 2:3]
    h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])
    if round:
        # project_positions_float = np.where(project_positions_float<0, 0, project_positions_float)
        # project_positions_float[:, 0] = np.where(project_positions_float[:, 0]>=w, w-1, project_positions_float[:, 0])
        # project_positions_float[:, 1] = np.where(project_positions_float[:, 1]>=h, h-1, project_positions_float[:, 1])
        project_positions_flat = np.round(project_positions_float).astype(np.short)
    else:
        project_positions_flat = np.floor(project_positions_float).astype(np.short)
        if not no_filter_frustum:
            h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])
            x_ge_0_index = np.where(project_positions_flat[:, 0] >= 0)
            project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
                x_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
            y_ge_0_index = np.where(project_positions_flat[:, 1] >= 0)
            project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
                y_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
            x_l_w_index = np.where(project_positions_flat[:, 0] < w)
            project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
                x_l_w_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
            y_l_h_index = np.where(project_positions_flat[:, 1] < h)
            project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
                y_l_h_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
            d_g_0_index = np.where(project_positions[:, 2] > 0)
            project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
                d_g_0_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)



    if k_ambiguity > 2:
        k_ambiguity_x_index = np.where(
            np.abs(project_positions_float[:, 0] - np.round(project_positions_float)[:, 0]) > 1 / k_ambiguity)
        project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
            k_ambiguity_x_index, project_positions_flat, project_positions_float, project_positions, points_start,
            colors)

        k_ambiguity_y_index = np.where(
            np.abs(project_positions_float[:, 1] - np.round(project_positions_float)[:, 1]) > 1 / k_ambiguity)
        project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
            k_ambiguity_y_index, project_positions_flat, project_positions_float, project_positions, points_start,
            colors)

    return project_positions_flat, project_positions_float, project_positions, points_start, colors  # , point_cloud


def find_2d_image(project_positions_flat, project_positions, points_start, colors, intrinsic_matrix,
                  need_second_img=False, no_filter_frustum=False):
    point_num = points_start.shape[0]

    # get color image
    h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])
    # print(h, w)
    image = np.ones((h, w, 3))
    dists = np.inf * np.ones((h, w))

    pixel_points = [[[] for j in range(w)] for i in range(h)]
    pixel_closest_point = [[-1 for j in range(w)] for i in range(h)]

    second_dists = np.inf * np.ones((h, w))

    # second_pixel_points = [ [ [] for j in range(w) ] for i in range(h) ]
    second_pixel_closest_point = [[-1 for j in range(w)] for i in range(h)]
    second_image = np.ones((h, w, 3))

    project_positions[:, :2] = project_positions_flat
    colored_positions = np.hstack([project_positions, colors]).astype(np.float16)
    # img_a = project_positions[:, :2] #image[project_positions[:, :1].astype("int").T[0], project_positions[:, 1:2].astype("int").T[0], :]
    unique = np.asarray(numpy.unique(np.asnumpy(project_positions_flat), axis=0))
    # unique = np.unique(project_positions_flat, axis=0)
    filtered_list = []
    # print(np.where(project_positions_flat == unique))
    # print("project_positions_flat", project_positions_flat.shape)
    # print("unique", unique.shape)
    # print("colored_positions", colored_positions.shape)
    # print(np.max(project_positions_flat),np.max(unique))

    unique = unique.astype(np.short)
    unique_ = np.repeat(unique[np.newaxis, :], project_positions_flat.shape[0], axis=0)
    project_positions_flat_ = np.repeat(project_positions_flat[:, np.newaxis, :], unique.shape[0], axis=1)
    colored_positions_ = np.repeat(colored_positions[:, np.newaxis, :], unique.shape[0], axis=1)
    # print("project_positions_flat", project_positions_flat_.dtype)
    # print("unique", unique_.dtype)
    # print("colored_positions", colored_positions_.dtype)
    same_positions_xy_index = (project_positions_flat_ == unique_)[:, :, 0] & (project_positions_flat_ == unique_)[:, :,
                                                                              1]
    depths_all = np.where(same_positions_xy_index, colored_positions_[:, :, 2], np.inf)

    filtered_positions = colored_positions_[np.argmin(depths_all, axis=0), np.arange(depths_all.shape[1])]
    '''
    for unique_item in unique:
        # print("unique_item", unique_item)
        # print("######", project_positions_flat)

        points_index_with_same_pixel_x = np.where(project_positions_flat[:, 0] == unique_item[0])[0]
        project_positions_flat_x = project_positions_flat[points_index_with_same_pixel_x]
        colored_positions_x = colored_positions[points_index_with_same_pixel_x]
        # print(unique_item[0], project_positions_flat_x, colored_positions_x)

        points_index_with_same_pixel_y = np.where(project_positions_flat_x[:, 1] == unique_item[1])[0]
        project_positions_flat_xy = project_positions_flat_x[points_index_with_same_pixel_y]
        colored_positions_xy = colored_positions_x[points_index_with_same_pixel_y]

        # print("$$$$$$$$$$$",unique_item, project_positions_flat[points_index_with_same_pixel][0])
        # print(b,unique_item,project_positions_flat.shape)
        # a = colored_positions[points_index_with_same_pixel] #numpy.unique(np.asnumpy(points_index_with_same_pixel), axis=0)
        # print("a", colored_positions_xy)
        depth = colored_positions_xy[:, 2]
        # print("depth", depth.shape)
        min_index = np.argmin(depth)
        # min_index_top_1, min_index_top_2 = np.argsort(depth)[0], np.argsort(depth)[1]

        filtered_list.append(colored_positions_xy[min_index])

        # assert depth.shape[0] == colored_positions_xy.shape[0]
        # assert colored_positions.shape[0] == project_positions_flat.shape[0]

    filtered_positions = np.array(filtered_list)
    # print(filtered_positions[:, :3])
    '''
    image[filtered_positions[:, 1:2].astype("int").T[0], filtered_positions[:, :1].astype("int").T[0],
    :] = filtered_positions[:, 3:6]
    # print("project_positions_flat", project_positions_flat.shape)
    # print("project_positions", project_positions.shape)
    # print("colored_positions", colored_positions.shape)
    # print("unique", unique.shape)
    # print("filtered_positions", filtered_positions.shape)

    if need_second_img:
        for i in range(point_num):

            x, y = np.asnumpy(project_positions_flat[i])
            dist = project_positions[i, 2]
            if y < 0 or x < 0 or x > w - 1 or y > h - 1 or dist <= 0:
                continue

            pixel_points[y][x].append(i)

            if dist < dists[y, x]:
                second_dists[y, x] = dists[y, x]
                second_pixel_closest_point[y][x] = pixel_closest_point[y][x]
                second_image[y, x] = image[y, x]

                dists[y, x] = dist
                pixel_closest_point[y][x] = i
                # print(y, x, i, colors[i], image[y, x])
                if no_filter_frustum:
                    image[y, x] = np.asarray(colors[i])
            elif dist < second_dists[y, x]:
                second_dists[y, x] = dist
                second_pixel_closest_point[y][x] = i
                second_image[y, x] = np.asarray(colors[i])

    return image, second_image, second_pixel_closest_point, pixel_points, pixel_closest_point


def find_outlier_rate(image, second_image, L1_threshold):
    h = image.shape[0]
    w = image.shape[1]
    bug = 0
    for r in range(w):
        for s in range(h):
            valid, consistent = check_consistency(image[s][r], second_image[s][r], L1_threshold)

            if valid and not consistent:
                # print(image[s][r])
                bug += 1
    # print(bug, w*h, bug/(w*h))
    return bug / (w * h)


def round_float(float_x):
    if np.abs(float_x - np.round(float_x)) < EPS_PIXEL:
        return np.round(float_x)
    else:
        return float_x


def find_k(project_positions_float, image, second_image, L1_threshold, second_pixel_closest_point, K):
    # # calculate k
    h = image.shape[0]
    w = image.shape[1]
    k_max = 3
    k_max_overflow_cnt = 0
    for r in range(w):
        # if r < 30: continue
        for s in range(h):
            valid, consistent = check_consistency(image[s][r], second_image[s][r], L1_threshold)
            if not valid or (valid and not consistent):
                continue
            pc = second_pixel_closest_point[s][r]
            k = k_max
            # project_positions_float[pc][0] = round_float(project_positions_float[pc][0])
            # project_positions_float[pc][1] = round_float(project_positions_float[pc][1])
            while not (k * r + 1 <= k * (project_positions_float[pc][0]) and k * (project_positions_float[pc][
                0]) <= k * r + k - 1 and k * s + 1 <= k * (project_positions_float[pc][1]) and k * (
                       project_positions_float[pc][
                           1]) <= k * s + k - 1):
                if k_max > K:
                    k_max = K
                    k_max_overflow_cnt += 1
                    break
                else:
                    k += 1.0
                    k_max = k

            # for k in range(3, 1000):
            #     # cover_center = False
            #     # for pc in pixel_points[s][r]:
            #     # print(pc, project_positions_float[pc])
            #     if k * r + 1 <= k * project_positions_float[pc][0] and k * project_positions_float[pc][
            #         0] <= k * r + k - 1 \
            #             and k * s + 1 <= k * project_positions_float[pc][1] and k * project_positions_float[pc][
            #         1] <= k * s + k - 1:
            #         # if check_consistency(colors[pc], colors[pixel_closest_point[s][r]], 50):
            #         #     print(r, s, k, pc)
            #         # cover_center=True
            #         # if cover_center:
            #         if k > k_max:
            #             k_max = k
            #         k_max_rs_list.append(k)
            #         break
            # print(r, s, k_max)

    return k_max, k_max_overflow_cnt / (h * w)


def check_exist_rho_inside_pixel(rho, axis_range, r, s):
    # print(rho[0](0), s, rho[1](0), r)
    # if np.abs(rho[0](0) - np.round(rho[0](0) )) < EPS:
    #     rho[0](0) = np.round(rho[0](0)
    # assert np.floor(round_float(rho[0](0))) == r, f"r: {np.floor(rho[0](0))}!={r}"
    # assert np.floor(round_float(rho[1](0))) == s, f"s: {np.floor(rho[1](0))}!={s}"
    # print(rho[0](range[0]), rho[0](range[1]), rho[1](range[0]), rho[1](range[1]), r, s)

    both_less_0_r = rho[0](axis_range[0]) - r < 0 and rho[0](axis_range[1]) - r < 0
    both_greater_0_r = rho[0](axis_range[0]) - r - 1 >= 0 and rho[0](axis_range[1]) - r - 1 >= 0
    both_less_0_s = rho[1](axis_range[0]) - s < 0 and rho[1](axis_range[1]) - s < 0
    both_greater_0_s = rho[1](axis_range[0]) - s - 1 >= 0 and rho[1](axis_range[1]) - s - 1 >= 0

    return not (both_less_0_s or both_greater_0_s or both_less_0_r or both_greater_0_r)


def calculate_C_delta(axis, delta, intrinsic_matrix, range, points_start, h, w, pixel_points):
    C_delta = 0
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    X = points_start[:, 0]
    Y = points_start[:, 1]
    Z = points_start[:, 2]
    b1 = range[0]
    b2 = range[1]
    if axis == "tz":
        print(f"np.max({delta} / ({Z} - {b2}))")
        C_delta = np.max(delta / (Z - b2))
    elif axis == "tx":
        C_delta = 0
    elif axis == "ty":
        C_delta = 0
    elif axis == "rz":
        C_delta = delta * max(fx/fy, fy/fx)
    elif axis == "rx":
        theta_b1 = np.maximum(delta * (fx * np.abs(X) + fy * np.abs(Y * np.cos(b1) + Z * np.sin(b1))) / (fy * (-Y * np.sin(b1) + Z * np.cos(b1)))
        , 2 * delta * np.abs(Y * np.cos(b1) + Z * np.sin(b1)) / (-Y * np.sin(b1) + Z * np.cos(b1)))
        theta_b2 = np.maximum(delta * (fx * np.abs(X) + fy * np.abs(Y * np.cos(b2) + Z * np.sin(b2))) / (fy * (-Y * np.sin(b2) + Z * np.cos(b2)))
        , 2 * delta * np.abs(Y * np.cos(b2) + Z * np.sin(b2)) / (-Y * np.sin(b2) + Z * np.cos(b2)))
        C_delta = delta**2 / fy + np.max(np.maximum(theta_b1, theta_b2))
    elif axis == "ry":
        theta_b1 = np.maximum(delta * (fy * np.abs(Y) + fx * np.abs(X * np.cos(b1) - Z * np.sin(b1))) / (fy * (X * np.sin(b1) + Z * np.cos(b1)))
        , 2 * delta * np.abs(X * np.cos(b1) - Z * np.sin(b1)) / (X * np.sin(b1) + Z * np.cos(b1)))
        theta_b2 = np.maximum(delta * (fy * np.abs(Y) + fx * np.abs(X * np.cos(b2) - Z * np.sin(b2))) / (fy * (X * np.sin(b2) + Z * np.cos(b2)))
        , 2 * delta * np.abs(X * np.cos(b2) - Z * np.sin(b2)) / (X * np.sin(b2) + Z * np.cos(b2)))
        C_delta = delta**2 / fx + np.max(np.maximum(theta_b1, theta_b2))
    return C_delta

# def find_sup_inf(intrinsic_matrix, tz_range, points_start, h, w, pixel_points, entire, k_ambiguity):
#     sup_U = np.ones([points_start.shape[0], w, h]) * tz_range[1]
#     inf_U = np.ones([points_start.shape[0], w, h]) * tz_range[0]
#     for r in range(w):
#         for s in range(h):
#             last_index = -1
#             for i in range(101):
#                 alpha_i = tz_range[0] + i * (tz_range[1] - tz_range[0]) / 100
#                 min_depth_index = np.argmin(points_start[np.array(pixel_points[s][r]), 2] - alpha_i)
#                 pc_index = pixel_points[s][r][min_depth_index]
#                 rho = [
#                     lambda tz: intrinsic_matrix[0][0] * points_start[pc_index, 0] / (
#                             points_start[pc_index, 2] - tz) +
#                                intrinsic_matrix[0][2],
#                     lambda tz: intrinsic_matrix[1][1] * points_start[pc_index, 1] / (
#                             points_start[pc_index, 2] - tz) +
#                                intrinsic_matrix[1][2]]
#                 if check_exist_rho_inside_pixel(rho, tz_range, r, s):
#                     if last_index != pc_index:
#                         if inf_U[pc_index][r][s] == tz_range[0] and sup_U[pc_index][r][s] == tz_range[1]:
#                             inf_U[pc_index][r][s] = alpha_i
#                             sup_U[pc_index][r][s] = alpha_i
#                         else:
#                             sup_U[pc_index][r][s] = alpha_i
#                     else:
#                         sup_U[pc_index][r][s] = alpha_i
#                 else:
#                     print("shit", i)
#     return sup_U, inf_U

def find_Delta_tz(intrinsic_matrix, tz_range, points_start, h, w, pixel_points, table_sup_inf, delta, entire, k_ambiguity):
    # sup_U, inf_U = find_sup_inf(intrinsic_matrix, tz_range, points_start, h, w, pixel_points, entire, k_ambiguity)
    L_points = np.maximum(
        intrinsic_matrix[0][0] * np.abs(points_start[:, 0]) / ((points_start[:, 2] - tz_range[1]) ** 2),
        intrinsic_matrix[1][1] * np.abs(points_start[:, 1]) / ((points_start[:, 2] - tz_range[1]) ** 2))
    if not entire:
        # delta = 0
        C_delta = calculate_C_delta("tz", delta, intrinsic_matrix, tz_range, points_start, h, w, pixel_points)
        # L_points = np.maximum(
        #     intrinsic_matrix[0][0] * np.abs(points_start[:, 0]) / ((points_start[:, 2] - tz_range[1]) ** 2),
        #     intrinsic_matrix[1][1] * np.abs(points_start[:, 1]) / ((points_start[:, 2] - tz_range[1]) ** 2))
        delta_L_rs_list = [1000000]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                table_rs_pc_list = [1000000]
                for pc_index in pixel_points[s][r]:
                    rho = [
                        lambda tz: intrinsic_matrix[0][0] * points_start[pc_index, 0] / (points_start[pc_index, 2] - tz) +
                                   intrinsic_matrix[0][2],
                        lambda tz: intrinsic_matrix[1][1] * points_start[pc_index, 1] / (points_start[pc_index, 2] - tz) +
                                   intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, tz_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index])
                        table_rs_pc_list.append(max(abs(rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                                     abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](table_sup_inf[pc_index][r][s][1]))))

                if min(table_rs_pc_list) - 2 * delta > 0 and min(table_rs_pc_list) != 1000000:
                    print(min(table_rs_pc_list))
                    delta_L_rs_list.append((1 / (max(L_rs_pc_list) + C_delta)) * (min(table_rs_pc_list) - 2 * delta))
        # print(1 / (max(L_rs_list) * k), k)
        return min(delta_L_rs_list)
    else:
        delta_rs_list = [10000]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                delta_rs_pc_list = [10000]
                for pc_index in pixel_points[s][r]:
                    rho = [
                        lambda tz: intrinsic_matrix[0][0] * points_start[pc_index, 0] / (
                                    points_start[pc_index, 2] - tz) +
                                   intrinsic_matrix[0][2],
                        lambda tz: intrinsic_matrix[1][1] * points_start[pc_index, 1] / (
                                    points_start[pc_index, 2] - tz) +
                                   intrinsic_matrix[1][2]]
                    if table_sup_inf[pc_index][r][s][0] == 100 or table_sup_inf[pc_index][r][s][1] == -100:
                        continue
                    # print("$$$$$$$$$$$$$$")
                    if check_exist_rho_inside_pixel(rho, tz_range, r, s):
                        # print(L_points[pc_index], min(abs(rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                        #                                              abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](table_sup_inf[pc_index][r][s][1]))))
                        delta_rs_pc_list.append(max(abs(rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                                     abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](table_sup_inf[pc_index][r][s][1]))) / L_points[pc_index])
                        # print("##############")
                        # rho[0](axis_range[0]) - r < 0
                        # table_consistent_interval[point_index][r][s][1] = alpha_i
                delta_rs_list.append(min(delta_rs_pc_list))
        # print(1 / (max(L_rs_list) * k), k)
        return min(delta_rs_list)


def find_Delta_tx(intrinsic_matrix, tx_range, points_start, h, w, pixel_points, table_sup_inf, delta, entire, k_ambiguity):
    L_points = intrinsic_matrix[0][0] / points_start[:, 2]
    if not entire:
        # delta = 0
        C_delta = calculate_C_delta("tx",delta, intrinsic_matrix, tx_range, points_start, h, w, pixel_points)
        delta_L_rs_list = [1000000]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                table_rs_pc_list = [1000000]
                for pc_index in pixel_points[s][r]:
                    rho = [
                        lambda tx: intrinsic_matrix[0][0] * (points_start[pc_index, 0] - tx) / points_start[
                            pc_index, 2] +
                                   intrinsic_matrix[0][2],
                        lambda tx: intrinsic_matrix[1][1] * points_start[pc_index, 1] / points_start[pc_index, 2] +
                                   intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, tx_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index])
                        table_rs_pc_list.append(max(abs(
                            rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                    abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](
                                                        table_sup_inf[pc_index][r][s][1]))))
                delta_L_rs_list.append((1 / (max(L_rs_pc_list) + C_delta)) * (min(table_rs_pc_list) - 2 * delta))
        # print(1 / (max(L_rs_list) * k), k)
        return min(delta_L_rs_list)
    else:
        # L_points = intrinsic_matrix[0][0] / points_start[:, 2]
        L_rs_list = [-1]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                for pc_index in pixel_points[s][r]:
                    rho = [
                        lambda tx: intrinsic_matrix[0][0] * (points_start[pc_index, 0] - tx) / points_start[pc_index, 2] +
                                   intrinsic_matrix[0][2],
                        lambda tx: intrinsic_matrix[1][1] * points_start[pc_index, 1] / points_start[pc_index, 2] +
                                   intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, tx_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index] / max(abs(rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                                     abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](table_sup_inf[pc_index][r][s][1]))))
                L_rs_list.append(max(L_rs_pc_list))
        # print(1 / (max(L_rs_list) * k), k)
        return 1 / max(L_rs_list)


def find_Delta_ty(intrinsic_matrix, ty_range, points_start, h, w, pixel_points,table_sup_inf, delta, entire, k_ambiguity):
    L_points = intrinsic_matrix[1][1] / points_start[:, 2]
    if not entire:
        # delta = 0
        C_delta = calculate_C_delta("ty",delta, intrinsic_matrix, ty_range, points_start, h, w, pixel_points)

        delta_L_rs_list = [1000000]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                table_rs_pc_list = [1000000]
                for pc_index in pixel_points[s][r]:
                    rho = [lambda ty: intrinsic_matrix[0][0] * points_start[pc_index, 0] / points_start[pc_index, 2] +
                                      intrinsic_matrix[0][2],
                           lambda ty: intrinsic_matrix[1][1] * (points_start[pc_index, 1] - ty) / points_start[
                               pc_index, 2] + intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, ty_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index])
                        table_rs_pc_list.append(max(abs(
                            rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                    abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](
                                                        table_sup_inf[pc_index][r][s][1]))))
                delta_L_rs_list.append((1 / (max(L_rs_pc_list) + C_delta)) * (min(table_rs_pc_list) - 2 * delta))
        # print(1 / (max(L_rs_list) * k), k)
        return min(delta_L_rs_list)
    else:
        # L_points = intrinsic_matrix[1][1] / points_start[:, 2]
        L_rs_list = [-1]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                for pc_index in pixel_points[s][r]:
                    rho = [lambda ty: intrinsic_matrix[0][0] * points_start[pc_index, 0] / points_start[pc_index, 2] +
                                      intrinsic_matrix[0][2],
                           lambda ty: intrinsic_matrix[1][1] * (points_start[pc_index, 1] - ty) / points_start[
                               pc_index, 2] + intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, ty_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index] / max(abs(rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                                     abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](table_sup_inf[pc_index][r][s][1]))))
                L_rs_list.append(max(L_rs_pc_list))
        # print(1 / (max(L_rs_list) * k), k)
        return 1 / max(L_rs_list)


def find_Delta_rz(intrinsic_matrix, rz_range, points_start, h, w, pixel_points,table_sup_inf, delta, entire, k_ambiguity):
    d_rho1 = intrinsic_matrix[0][0] * np.sqrt(points_start[:, 0] ** 2 + points_start[:, 1] ** 2) / points_start[:, 2]
    d_rho1 = np.where(-points_start[:, 0] / points_start[:, 1] < np.tan(rz_range[0]), intrinsic_matrix[0][0] * np.abs(
        points_start[:, 0] * (-np.sin(rz_range[0])) + points_start[:, 1] * np.cos(rz_range[0])) / points_start[:, 2],
                      d_rho1)
    d_rho1 = np.where(-points_start[:, 0] / points_start[:, 1] > np.tan(rz_range[1]), intrinsic_matrix[0][0] * np.abs(
        points_start[:, 0] * (-np.sin(rz_range[1])) + points_start[:, 1] * np.cos(rz_range[1])) / points_start[:, 2],
                      d_rho1)

    d_rho2 = intrinsic_matrix[1][1] * np.sqrt(points_start[:, 0] ** 2 + points_start[:, 1] ** 2) / points_start[:, 2]
    d_rho2 = np.where(-points_start[:, 0] / points_start[:, 1] < np.tan(rz_range[0]), intrinsic_matrix[1][1] * np.abs(
        points_start[:, 1] * (-np.sin(rz_range[0])) + points_start[:, 0] * np.cos(rz_range[0])) / points_start[:, 2],
                      d_rho2)
    d_rho2 = np.where(-points_start[:, 0] / points_start[:, 1] > np.tan(rz_range[1]), intrinsic_matrix[1][1] * np.abs(
        points_start[:, 1] * (-np.sin(rz_range[1])) + points_start[:, 0] * np.cos(rz_range[1])) / points_start[:, 2],
                      d_rho2)

    # L_points = np.maximum(intrinsic_matrix[0][0]*np.sqrt(points_start[:, 0]**2 + points_start[:, 1]**2)/points_start[:, 2],
    #                       intrinsic_matrix[1][1]*np.sqrt(points_start[:, 0]**2 + points_start[:, 1]**2)/points_start[:, 2])

    L_points = np.maximum(d_rho1, d_rho2)
    if not entire:
        # delta = 0
        C_delta = calculate_C_delta("rz",delta, intrinsic_matrix, rz_range, points_start, h, w, pixel_points)
        delta_L_rs_list = [1000000]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                table_rs_pc_list = [1000000]
                for pc_index in pixel_points[s][r]:
                    rho = [lambda rz: intrinsic_matrix[0][0] * (
                            points_start[pc_index, 0] * np.cos(rz) + points_start[pc_index, 1] * np.sin(rz)) /
                                      points_start[pc_index, 2] + intrinsic_matrix[0][2],
                           lambda rz: intrinsic_matrix[1][1] * (
                                   points_start[pc_index, 1] * np.cos(rz) + points_start[pc_index, 0] * np.sin(rz)) /
                                      points_start[pc_index, 2] + intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, rz_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index])
                        table_rs_pc_list.append(max(abs(
                            rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                    abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](
                                                        table_sup_inf[pc_index][r][s][1]))))
                delta_L_rs_list.append((1 / (max(L_rs_pc_list) + C_delta)) * (min(table_rs_pc_list) - 2 * delta))
        # print(1 / (max(L_rs_list) * k), k)
        return min(delta_L_rs_list)
    else:
        L_rs_list = [-1]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                for pc_index in pixel_points[s][r]:
                    rho = [lambda rz: intrinsic_matrix[0][0] * (
                            points_start[pc_index, 0] * np.cos(rz) + points_start[pc_index, 1] * np.sin(rz)) /
                                      points_start[pc_index, 2] + intrinsic_matrix[0][2],
                           lambda rz: intrinsic_matrix[1][1] * (
                                   points_start[pc_index, 1] * np.cos(rz) + points_start[pc_index, 0] * np.sin(rz)) /
                                      points_start[pc_index, 2] + intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, rz_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index] / max(abs(rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                                     abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](table_sup_inf[pc_index][r][s][1]))))
                L_rs_list.append(max(L_rs_pc_list))
        # print(1 / (max(L_rs_list) * k), k)
        return 1 / max(L_rs_list)


def find_Delta_ry(intrinsic_matrix, ry_range, points_start, h, w, pixel_points,table_sup_inf, delta, entire, k_ambiguity):
    # C_delta = 0
    # if not entire:
    #     C_delta = calculate_C_delta("tz", intrinsic_matrix, ry_range, points_start, h, w, pixel_points)
    d_rho1 = np.maximum(intrinsic_matrix[0][0] * (points_start[:, 0] ** 2 + points_start[:, 2] ** 2) / ((points_start[:,
                                                                                                         0] * np.sin(
        ry_range[0]) + points_start[:, 2] * np.cos(ry_range[0])) ** 2),
                        intrinsic_matrix[0][0] * (points_start[:, 0] ** 2 + points_start[:, 2] ** 2) / ((points_start[:,
                                                                                                         0] * np.sin(
                            ry_range[1]) + points_start[:, 2] * np.cos(ry_range[1])) ** 2))

    d_rho2 = np.maximum(intrinsic_matrix[1][1] * np.abs(
        points_start[:, 1] * (points_start[:, 0] * np.cos(ry_range[0]) - points_start[:, 2] * np.sin(ry_range[0]))) / ((
                                                                                                                               points_start[
                                                                                                                               :,
                                                                                                                               0] * np.sin(
                                                                                                                           ry_range[
                                                                                                                               0]) + points_start[
                                                                                                                                     :,
                                                                                                                                     2] * np.cos(
                                                                                                                           ry_range[
                                                                                                                               0])) ** 2),
                        intrinsic_matrix[1][1] * points_start[:, 1] * np.abs(
                            points_start[:, 0] * np.cos(ry_range[1]) - points_start[:, 2] * np.sin(ry_range[1])) / ((
                                                                                                                            points_start[
                                                                                                                            :,
                                                                                                                            0] * np.sin(
                                                                                                                        ry_range[
                                                                                                                            1]) + points_start[
                                                                                                                                  :,
                                                                                                                                  2] * np.cos(
                                                                                                                        ry_range[
                                                                                                                            1])) ** 2))

    # L_points = np.maximum(intrinsic_matrix[0][0]*np.sqrt(points_start[:, 0]**2 + points_start[:, 1]**2)/points_start[:, 2],
    #                       intrinsic_matrix[1][1]*np.sqrt(points_start[:, 0]**2 + points_start[:, 1]**2)/points_start[:, 2])

    L_points = np.maximum(d_rho1, d_rho2)
    if not entire:
        # delta = 0
        C_delta = calculate_C_delta("ry",delta, intrinsic_matrix, ry_range, points_start, h, w, pixel_points)
        delta_L_rs_list = [1000000]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                table_rs_pc_list = [1000000]
                for pc_index in pixel_points[s][r]:
                    rho = [lambda ry: intrinsic_matrix[0][0] * (
                            points_start[pc_index, 0] * np.cos(ry) - points_start[pc_index, 2] * np.sin(ry)) /
                                      (points_start[pc_index, 0] * np.sin(ry) + points_start[pc_index, 2] * np.cos(
                                          ry)) +
                                      intrinsic_matrix[0][2],
                           lambda ry: intrinsic_matrix[1][1] * points_start[pc_index, 1] /
                                      (points_start[pc_index, 0] * np.sin(ry) + points_start[pc_index, 2] * np.cos(
                                          ry)) +
                                      intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, ry_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index])
                        table_rs_pc_list.append(max(abs(
                            rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                    abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](
                                                        table_sup_inf[pc_index][r][s][1]))))
                delta_L_rs_list.append((1 / (max(L_rs_pc_list) + C_delta)) * (min(table_rs_pc_list) - 2 * delta))
        # print(1 / (max(L_rs_list) * k), k)
        return min(delta_L_rs_list)
    else:
        L_rs_list = [-1]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                for pc_index in pixel_points[s][r]:
                    rho = [lambda ry: intrinsic_matrix[0][0] * (
                            points_start[pc_index, 0] * np.cos(ry) - points_start[pc_index, 2] * np.sin(ry)) /
                                      (points_start[pc_index, 0] * np.sin(ry) + points_start[pc_index, 2] * np.cos(ry)) +
                                      intrinsic_matrix[0][2],
                           lambda ry: intrinsic_matrix[1][1] * points_start[pc_index, 1] /
                                      (points_start[pc_index, 0] * np.sin(ry) + points_start[pc_index, 2] * np.cos(ry)) +
                                      intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, ry_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index] / max(abs(rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                                     abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](table_sup_inf[pc_index][r][s][1]))))
                L_rs_list.append(max(L_rs_pc_list))
        # print(1 / (max(L_rs_list) * k), k)
        return 1 / max(L_rs_list)


def find_Delta_rx(intrinsic_matrix, rx_range, points_start, h, w, pixel_points, table_sup_inf, delta,entire, k_ambiguity):
    # C_delta = 0
    # if not entire:
    #     C_delta = calculate_C_delta("tz", intrinsic_matrix, rx_range, points_start, h, w, pixel_points)
    d_rho1 = np.maximum(intrinsic_matrix[0][0] * np.abs(
        points_start[:, 0] * (points_start[:, 1] * np.cos(rx_range[0]) + points_start[:, 2] * np.sin(rx_range[0]))) / ((
                                                                                                                               -points_start[
                                                                                                                                :,
                                                                                                                                1] * np.sin(
                                                                                                                           rx_range[
                                                                                                                               0]) + points_start[
                                                                                                                                     :,
                                                                                                                                     2] * np.cos(
                                                                                                                           rx_range[
                                                                                                                               0])) ** 2),
                        intrinsic_matrix[0][0] * np.abs(points_start[:, 0] * (
                                points_start[:, 1] * np.cos(rx_range[1]) + points_start[:, 2] * np.sin(
                            rx_range[1]))) / ((-points_start[:, 1] * np.sin(rx_range[1]) + points_start[:,
                                                                                           2] * np.cos(
                            rx_range[1])) ** 2))

    d_rho2 = np.maximum(
        intrinsic_matrix[1][1] * (points_start[:, 1] ** 2 + points_start[:, 2] ** 2) / ((-points_start[:,
                                                                                          1] * np.sin(
            rx_range[0]) + points_start[:, 2] * np.cos(rx_range[0])) ** 2),
        intrinsic_matrix[1][1] * (points_start[:, 1] ** 2 + points_start[:, 2] ** 2) / ((-points_start[:,
                                                                                          1] * np.sin(
            rx_range[1]) + points_start[:, 2] * np.cos(rx_range[1])) ** 2))

    # L_points = np.maximum(intrinsic_matrix[0][0]*np.sqrt(points_start[:, 0]**2 + points_start[:, 1]**2)/points_start[:, 2],
    #                       intrinsic_matrix[1][1]*np.sqrt(points_start[:, 0]**2 + points_start[:, 1]**2)/points_start[:, 2])

    L_points = np.maximum(d_rho1, d_rho2)
    if not entire:
        # delta = 0
        C_delta = calculate_C_delta("rx",delta, intrinsic_matrix, rx_range, points_start, h, w, pixel_points)
        delta_L_rs_list = [1000000]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                table_rs_pc_list = [-1]
                for pc_index in pixel_points[s][r]:
                    rho = [lambda rx: intrinsic_matrix[0][0] * points_start[pc_index, 0] /
                                      (-points_start[pc_index, 1] * np.sin(rx) + points_start[pc_index, 2] * np.cos(
                                          rx)) +
                                      intrinsic_matrix[0][2],
                           lambda rx: intrinsic_matrix[1][1] * (
                                   points_start[pc_index, 1] * np.cos(rx) + points_start[pc_index, 2] * np.sin(rx)) /
                                      (-points_start[pc_index, 1] * np.sin(rx) + points_start[pc_index, 2] * np.cos(
                                          rx)) +
                                      intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, rx_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index])
                        table_rs_pc_list.append(max(abs(
                            rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                    abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](
                                                        table_sup_inf[pc_index][r][s][1]))))
                delta_L_rs_list.append((1 / (max(L_rs_pc_list) + C_delta)) * (min(table_rs_pc_list) - 2 * delta))
        # print(1 / (max(L_rs_list) * k), k)
        return min(delta_L_rs_list)
    else:
        L_rs_list = [-1]
        # print(len(pixel_points), h, w)
        for r in range(w):
            for s in range(h):
                L_rs_pc_list = [-1]
                for pc_index in pixel_points[s][r]:
                    rho = [lambda rx: intrinsic_matrix[0][0] * points_start[pc_index, 0] /
                                      (-points_start[pc_index, 1] * np.sin(rx) + points_start[pc_index, 2] * np.cos(rx)) +
                                      intrinsic_matrix[0][2],
                           lambda rx: intrinsic_matrix[1][1] * (
                                   points_start[pc_index, 1] * np.cos(rx) + points_start[pc_index, 2] * np.sin(rx)) /
                                      (-points_start[pc_index, 1] * np.sin(rx) + points_start[pc_index, 2] * np.cos(rx)) +
                                      intrinsic_matrix[1][2]]
                    if check_exist_rho_inside_pixel(rho, rx_range, r, s):
                        L_rs_pc_list.append(L_points[pc_index] / max(abs(rho[0](table_sup_inf[pc_index][r][s][0]) - rho[0](table_sup_inf[pc_index][r][s][1])),
                                                                     abs(rho[1](table_sup_inf[pc_index][r][s][0]) - rho[1](table_sup_inf[pc_index][r][s][1]))))
                L_rs_list.append(max(L_rs_pc_list))
        # print(1 / (max(L_rs_list) * k), k)
        return 1 / max(L_rs_list)


def find_Delta(axis, axis_range, intrinsic_matrix, points_start, image_shape, pixel_points, table_sup_inf, delta,  entire, k_ambiguity):
    h = image_shape[0]
    w = image_shape[1]
    if axis == 'tz':
        Delta = find_Delta_tz(intrinsic_matrix, axis_range, points_start, h, w, pixel_points,table_sup_inf, delta, entire=entire,
                              k_ambiguity=k_ambiguity)
        print('tz', np.mean(points_start, axis=0), axis_range, Delta, (axis_range[1] - axis_range[0]) / Delta)
    elif axis == 'tx':
        Delta = find_Delta_tx(intrinsic_matrix, axis_range, points_start, h, w, pixel_points,table_sup_inf, delta, entire=entire,
                              k_ambiguity=k_ambiguity)
        print('tx', np.mean(points_start, axis=0), axis_range, Delta, (axis_range[1] - axis_range[0]) / Delta)
    elif axis == 'ty':
        Delta = find_Delta_ty(intrinsic_matrix, axis_range, points_start, h, w, pixel_points,table_sup_inf, delta, entire=entire,
                              k_ambiguity=k_ambiguity)
        print('ty', np.mean(points_start, axis=0), axis_range, Delta, (axis_range[1] - axis_range[0]) / Delta)
    elif axis == 'rz':
        Delta = find_Delta_rz(intrinsic_matrix, axis_range, points_start, h, w, pixel_points,table_sup_inf, delta, entire=entire,
                              k_ambiguity=k_ambiguity)
        print('rz', np.mean(points_start, axis=0), axis_range, Delta, (axis_range[1] - axis_range[0]) / Delta)
    elif axis == 'ry':
        Delta = find_Delta_ry(intrinsic_matrix, axis_range, points_start, h, w, pixel_points,table_sup_inf, delta, entire=entire,
                              k_ambiguity=k_ambiguity)
        print('ry', np.mean(points_start, axis=0), axis_range, Delta, (axis_range[1] - axis_range[0]) / Delta)
    else:
        Delta = find_Delta_rx(intrinsic_matrix, axis_range, points_start, h, w, pixel_points,table_sup_inf, delta, entire=entire,
                              k_ambiguity=k_ambiguity)
        print('rx', np.mean(points_start, axis=0), axis_range, Delta, (axis_range[1] - axis_range[0]) / Delta)
    return Delta


def find_new_extrinsic_matrix(extrinsic_matrix, axis, alpha):
    alpha = np.asnumpy(alpha)
    if axis == 'tz':
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = np.array([[0, 0, alpha]])
    elif axis == 'tx':
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = np.array([[alpha, 0, 0]])
    elif axis == 'ty':
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = np.array([[0, alpha, 0]])
    elif axis == 'rz':
        R = np.array([[numpy.cos(alpha), -numpy.sin(alpha), 0],
                      [numpy.sin(alpha), numpy.cos(alpha), 0],
                      [0, 0, 1]])
        t = np.array([[0, 0, 0]])
    elif axis == 'ry':
        R = np.array([[numpy.cos(alpha), 0, numpy.sin(alpha)],
                      [0, 1, 0],
                      [-numpy.sin(alpha), 0, numpy.cos(alpha)]])
        t = np.array([[0, 0, 0]])
    else:
        R = np.array([[1, 0, 0],
                      [0, numpy.cos(alpha), -numpy.sin(alpha)],
                      [0, numpy.sin(alpha), numpy.cos(alpha)]])
        t = np.array([[0, 0, 0]])
    rel_matrix = np.vstack((np.hstack((R, t.T)), np.array([0, 0, 0, 1])))
    return extrinsic_matrix @ rel_matrix


def find_next_neighbor_square(image_all):
    print(image_all.shape)
    neighbor_diff = np.zeros(image_all.shape)[1:, :, :, :]
    shit_num = 0
    max_error = 0
    for n in range(image_all.shape[0] - 1):
        for k in range(image_all.shape[3]):
            for i in range(image_all.shape[1]):
                for j in range(image_all.shape[2]):
                    i_j_diff = [np.array(0)]
                    for i_ in [-1, 0, 1]:
                        for j_ in [-1, 0, 1]:
                            if i + i_ < 0:
                                i_ = 0
                            elif i + i_ >= image_all.shape[1]:
                                i_ = 0
                            if j + j_ < 0:
                                j_ = 0
                            elif j + j_ >= image_all.shape[2]:
                                j_ = 0
                            i_j__ = []
                            for i__ in [-1, 0, 1]:
                                for j__ in [-1, 0, 1]:
                                    if i + i__ < 0:
                                        i__ = 0
                                    elif i + i__ >= image_all.shape[1]:
                                        i__ = 0
                                    if j + j__ < 0:
                                        j__ = 0
                                    elif j + j__ >= image_all.shape[2]:
                                        j__ = 0
                                    i_j__.append(image_all[n + 1, i + i__, j + j__, k])
                            i_j__ = np.stack(i_j__)
                            if np.min(np.abs(image_all[n, i + i_, j + j_, k] - i_j__)) == 0:
                                i_j_diff.append(np.square(image_all[n, i, j, k] - image_all[n, i + i_, j + j_, k]))

                    if len(i_j_diff) == 1:
                        if max_error < np.min(np.abs(image_all[n, i + i_, j + j_, k] - i_j__)):
                            max_error = np.min(np.abs(image_all[n, i + i_, j + j_, k] - i_j__))
                        # print()
                        # print("forward shit", n, i, j, k)
                        shit_num += 1
                        # i_j_diff = [[0]]
                    # print(i_j_diff)
                    i_j_diff = np.stack(i_j_diff)
                    # print(neighbor_diff[:, i, j, :])
                    neighbor_diff[n, i, j, k] = np.amax(i_j_diff, axis=0)

            # print(neighbor_diff[:, i, j, :])
    # x = [image_all[:, i, j, :] - image_all[:, (i+i_)%image_all.shape[1], (j+j_)%image_all.shape[1], :]
    #      for i_ in [-1, 0, 1] for j_ in [-1, 0, 1] for i in range(image_all.shape[1]) for j in range(image_all.shape[2])]
    # print(neighbor_diff)
    print(f"forward: shit_num: {shit_num}/{image_all.shape}, max_error: {max_error} ")
    return neighbor_diff


def find_last_neighbor_square(image_all):
    print(image_all.shape)
    neighbor_diff = np.zeros(image_all.shape)[1:, :, :, :]
    shit_num = 0
    max_error = 0
    for n in range(image_all.shape[0] - 1):
        for k in range(image_all.shape[3]):
            for i in range(image_all.shape[1]):
                for j in range(image_all.shape[2]):
                    i_j_diff = [np.array(0)]
                    for i_ in [-1, 0, 1]:
                        for j_ in [-1, 0, 1]:
                            if i + i_ < 0:
                                i_ = 0
                            elif i + i_ >= image_all.shape[1]:
                                i_ = 0
                            if j + j_ < 0:
                                j_ = 0
                            elif j + j_ >= image_all.shape[2]:
                                j_ = 0
                            i_j__ = []
                            for i__ in [-1, 0, 1]:
                                for j__ in [-1, 0, 1]:
                                    if i + i__ < 0:
                                        i__ = 0
                                    elif i + i__ >= image_all.shape[1]:
                                        i__ = 0
                                    if j + j__ < 0:
                                        j__ = 0
                                    elif j + j__ >= image_all.shape[2]:
                                        j__ = 0
                                    i_j__.append(image_all[n, i + i__, j + j__, k])
                            i_j__ = np.stack(i_j__)
                            if np.min(np.abs(image_all[n + 1, i + i_, j + j_, k] - i_j__)) == 0:
                                # if np.min(image_all[n+1, i + i_, j + j_, k] - np.array([[image_all[n, i - 1, j - 1, k],
                                #                                                        image_all[n, i - 1, j, k],
                                #                                                        image_all[n, i - 1, j + 1, k]],
                                #                                                       [image_all[n, i, j - 1, k],
                                #                                                        image_all[n, i, j, k],
                                #                                                        image_all[n, i, j + 1, k]],
                                #                                                       [image_all[n, i + 1, j - 1, k],
                                #                                                        image_all[n, i + 1, j, k],
                                #                                                        image_all[
                                #                                                            n, i + 1, j + 1, k]]])) < 0.01:
                                i_j_diff.append(
                                    np.square(image_all[n + 1, i, j, k] - image_all[n + 1, i + i_, j + j_, k]))
                    # if len(i_j_diff) == 0:
                    #     print("back shit", n+1, i, j, k)
                    #     i_j_diff = [0]
                    if len(i_j_diff) == 1:
                        if max_error < np.min(np.abs(image_all[n + 1, i + i_, j + j_, k] - i_j__)):
                            max_error = np.min(np.abs(image_all[n + 1, i + i_, j + j_, k] - i_j__))
                        # print()
                        # print("forward shit", n, i, j, k)
                        shit_num += 1
                    i_j_diff = np.stack(i_j_diff)
                    # print(neighbor_diff[:, i, j, :])
                    neighbor_diff[n, i, j, k] = np.amax(i_j_diff, axis=0)
            # print(neighbor_diff[:, i, j, :])
    # x = [image_all[:, i, j, :] - image_all[:, (i+i_)%image_all.shape[1], (j+j_)%image_all.shape[1], :]
    #      for i_ in [-1, 0, 1] for j_ in [-1, 0, 1] for i in range(image_all.shape[1]) for j in range(image_all.shape[2])]
    # print(neighbor_diff)
    print(f"backward: shit_num: {shit_num}/{image_all.shape}, max_error: {max_error} ")
    return neighbor_diff
    # return np.square(image_all.reshape(image_all.shape[0], -1)[:-1, :] - image_all.reshape(image_all.shape[0], -1)[1:, :])

def check_monotonicity(table_consistent_points, r, s):
    table_array = np.array(table_consistent_points)
    alpha_array = table_array[:, r, s].flatten()
    u, indices, counts = np.unique(alpha_array, return_index=True, return_counts=True)
    return (np.sort(indices)[1:] == np.sort(indices+counts)[: -1]).all()
    # last_item = alpha_array[0]
    # for i in range(alpha_array.shape[0]):
    #     if last_item

def find_table_sup_inf(complete_3D_oracle, extrinsic_matrix, intrinsic_matrix, axis, axis_range, image_shape):
    # table_consistent_interval
    h = image_shape[0]
    w = image_shape[1]
    table_consistent_points = [[[-1 for j in range(h)] for i in range(w)] for k in range(RESOLUTION)]
    for slide_i in trange(RESOLUTION):

        alpha_i = axis_range[0] + slide_i * (axis_range[1] - axis_range[0]) / RESOLUTION
        extrinsic_matrix_i = find_new_extrinsic_matrix(extrinsic_matrix, axis, alpha_i)
        project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
            complete_3D_oracle, extrinsic_matrix_i, intrinsic_matrix, k_ambiguity=-1, no_filter_frustum=True)  # k_ambiguity)
        image, second_image, second_pixel_closest_point, pixel_points, pixel_closest_point = find_2d_image(project_positions_flat, project_positions, points_start, colors,
                                         intrinsic_matrix, need_second_img=True, no_filter_frustum=True)
        # print(points_start.shape[0])
        assert complete_3D_oracle.shape[0] == points_start.shape[0]
        # matplotlib.image.imsave(f"test_{slide_i}.png", image.get())
        # project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
        #     complete_3D_oracle, extrinsic_matrix, intrinsic_matrix, k_ambiguity=-1)  # k_ambiguity)
        # # print("start find 2d image")
        # image, second_image, second_pixel_closest_point, pixel_points = find_2d_image(project_positions_flat,
        #                                                                               project_positions, points_start,
        #                                                                               colors, intrinsic_matrix,
        #                                                                               need_second_img=True)

        for r in range(w):
            for s in range(h):
                table_consistent_points[slide_i][r][s] = pixel_closest_point[s][r]
    # np.save("table_consistent_points.npy", table_consistent_points)
    table_consistent_interval = [[[[100, -100] for j in range(h)] for i in range(w)] for k in range(points_start.shape[0])]
    for r in range(w):
        for s in range(h):
            print(r, s)
            # check monotonicity
            if not check_monotonicity(table_consistent_points, r, s):
                print("not mono", r, s)
            for slide_i in range(RESOLUTION):
                alpha_i = axis_range[0] + slide_i * (axis_range[1] - axis_range[0]) / RESOLUTION
                point_index = table_consistent_points[slide_i][r][s]
                # print(point_index, len(table_consistent_interval))
                # print(r, len(table_consistent_interval[point_index]))
                # print(s, len(table_consistent_interval[point_index][r]))
                # print(len(table_consistent_interval[point_index][r][s]))
                if table_consistent_interval[point_index][r][s][1] == -100:
                    table_consistent_interval[point_index][r][s][1] = alpha_i
                    # print("table_consistent_interval[point_index][r][s][1]", slide_i)
                else:
                    table_consistent_interval[point_index][r][s][0] = alpha_i
                    # print("table_consistent_interval[point_index][r][s][0]", slide_i)

                # L_rs_pc_list = [-1]
                # for pc_index in pixel_points[s][r]:
                #     rho = [lambda rx: intrinsic_matrix[0][0] * points_start[pc_index, 0] /
                #                       (-points_start[pc_index, 1] * np.sin(rx) + points_start[pc_index, 2] * np.cos(rx)) +
                #                       intrinsic_matrix[0][2],
                #            lambda rx: intrinsic_matrix[1][1] * (
                #                    points_start[pc_index, 1] * np.cos(rx) + points_start[pc_index, 2] * np.sin(rx)) /
                #                       (-points_start[pc_index, 1] * np.sin(rx) + points_start[pc_index, 2] * np.cos(rx)) +
                #                       intrinsic_matrix[1][2]]
                #     if check_exist_rho_inside_pixel(rho, rx_range, r, s):
                #         L_rs_pc_list.append(L_points[pc_index])
                # L_rs_list.append(max(L_rs_pc_list))

    return numpy.array(table_consistent_interval)

def calculate_delta(complete_3D_oracle, one_frame_pc, extrinsic_matrix, intrinsic_matrix, axis, axis_range, image_shape):
    # h = image_shape[0]
    # w = image_shape[1]
    # table_consistent_points = [[[-1 for j in range(h)] for i in range(w)] for k in range(100)]
    delta_list = []
    print(f"starting using RESOLUTION: {RESOLUTION}")
    for slide_i in range(RESOLUTION):

        alpha_i = axis_range[0] + slide_i * (axis_range[1] - axis_range[0]) / RESOLUTION
        extrinsic_matrix_i = find_new_extrinsic_matrix(extrinsic_matrix, axis, alpha_i)
        project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
            complete_3D_oracle, extrinsic_matrix_i, intrinsic_matrix, k_ambiguity=-1)  # k_ambiguity)
        project_positions_flat_one, project_positions_float_one, project_positions_one, points_start_one, colors_one = projection_oracle(
            one_frame_pc, extrinsic_matrix_i, intrinsic_matrix, k_ambiguity=-1)
        # image, second_image, second_pixel_closest_point, pixel_points, pixel_closest_point = find_2d_image(project_positions_flat, project_positions, points_start, colors,
        #                                  intrinsic_matrix, need_second_img=True)
        delta_list_one = []
        # print(points_start_one.shape[0])
        for i in range(points_start_one.shape[0]):

            delta_fixed_one = np.maximum(np.abs(project_positions_float[:, 0] - project_positions_float_one[i, 0]),
                       np.abs(project_positions_float[:, 1] - project_positions_float_one[i, 1]))
            # print(f"delta_fixed_one: {delta_fixed_one.shape}")
            invalid_depth_index = project_positions[:, 2] < project_positions_one[i, 2]
            # print(invalid_depth_index.shape)
            delta_fixed_one[invalid_depth_index] = 10000
            delta_list_one.append(delta_fixed_one)
        delta_array = np.array(delta_list_one)
        delta_array = np.amin(delta_array, axis=0)
        delta_array = np.asnumpy(delta_array)
        modified_array = numpy.delete(delta_array, numpy.where(delta_array == 10000))
        # delta = np.amax(np.amin(delta_array, axis=0))
        modified_array = np.asarray(modified_array)
        print(np.percentile(modified_array, 99))
        # delta = np.amax(modified_array)
        delta = np.amax(np.percentile(modified_array, 99))
        delta_list.append(delta)
    return max(delta_list)

                # intrinsic_matrix[0][0] * np.abs(points_start[:, 0]) / ((points_start[:, 2] - tz_range[1]) ** 2),
                # intrinsic_matrix[1][1] * np.abs(points_start[:, 1]) / ((points_start[:, 2] - tz_range[1]) ** 2))
            # max(project_positions_float[:, 0] - project_positions_float_one[i, 0], project_positions_float[:, 1] - project_positions_float_one[i, 1])
            # project_positions_float_one[i] -

        # project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
        #     complete_3D_oracle, extrinsic_matrix, intrinsic_matrix, k_ambiguity=-1)  # k_ambiguity)
        # # print("start find 2d image")
        # image, second_image, second_pixel_closest_point, pixel_points = find_2d_image(project_positions_flat,
        #                                                                               project_positions, points_start,
        #                                                                               colors, intrinsic_matrix,
        #                                                                               need_second_img=True)
    #
    #     for r in range(w):
    #         for s in range(h):
    #             table_consistent_points[slide_i][r][s] = pixel_closest_point[s][r]
    # table_consistent_interval = [[[[100, -100] for j in range(h)] for i in range(w)] for k in range(points_start.shape[0])]
    # for r in range(w):
    #     for s in range(h):
    #         # check monotonicity
    #         if not check_monotonicity(table_consistent_points, r, s):
    #             print("not mono", r, s)
    #         for slide_i in range(100):
    #             alpha_i = axis_range[0] + slide_i * (axis_range[1] - axis_range[0]) / (
    #                     np.floor((axis_range[1] - axis_range[0]) / Delta) + 1)
    #             point_index = table_consistent_points[slide_i][r][s]
    #             # check monotonicity
    #             if table_consistent_interval[point_index][r][s][1] == -100:
    #                 table_consistent_interval[point_index][r][s][1] = alpha_i
    #             else:
    #                 table_consistent_interval[point_index][r][s][0] = alpha_i
    #
    #             # L_rs_pc_list = [-1]
    #             # for pc_index in pixel_points[s][r]:
    #             #     rho = [lambda rx: intrinsic_matrix[0][0] * points_start[pc_index, 0] /
    #             #                       (-points_start[pc_index, 1] * np.sin(rx) + points_start[pc_index, 2] * np.cos(rx)) +
    #             #                       intrinsic_matrix[0][2],
    #             #            lambda rx: intrinsic_matrix[1][1] * (
    #             #                    points_start[pc_index, 1] * np.cos(rx) + points_start[pc_index, 2] * np.sin(rx)) /
    #             #                       (-points_start[pc_index, 1] * np.sin(rx) + points_start[pc_index, 2] * np.cos(rx)) +
    #             #                       intrinsic_matrix[1][2]]
    #             #     if check_exist_rho_inside_pixel(rho, rx_range, r, s):
    #             #         L_rs_pc_list.append(L_points[pc_index])
    #             # L_rs_list.append(max(L_rs_pc_list))
    #
    # return np.asarray(table_consistent_interval)


if __name__ == '__main__':
    torch.set_num_threads(2)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split, args.transtype, not_entire=args.not_entire)

    # init transformers
    # diff_projection = DiffResolvableProjectionTransformer(dataset[0][0], args.transtype[-2:])

    # modify outfile name to distinguish different parts
    # if args.start != 0 or args.max != -1:
    #     args.aliasfile += f'_start_{args.start}_end_{args.max}'

    # setproctitle.setproctitle(f'rotation_aliasing_{args.dataset}from{args.start}to{args.max}')
    k_ambiguity = args.k_ambiguity
    if not os.path.exists(os.path.dirname(args.aliasfile)):
        os.makedirs(os.path.dirname(args.aliasfile))
    f = open(args.aliasfile, 'w')
    print('no.\tmaxl2sqr\tnum_slice\ttime', file=f, flush=True)

    before_time = time()

    for i in range(len(dataset)):

        if i < args.start:
            continue

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i >= args.max >= 0:
            break

        (x, label) = dataset[i]

        print('working on #', i)
        global_before_time = time()
        intrinsic_matrix = x["intrinsic_matrix"]
        intrinsic_matrix[0][0] /= 8
        intrinsic_matrix[1][1] /= 8
        intrinsic_matrix[0][2] /= 8
        intrinsic_matrix[1][2] /= 8
        if args.small_img:
            intrinsic_matrix[0][0] = intrinsic_matrix[0][0] * 2 / 5
            intrinsic_matrix[1][1] = intrinsic_matrix[1][1] * 4 / 9
            intrinsic_matrix[0][2] = intrinsic_matrix[0][2] * 2 / 5 - 4
            intrinsic_matrix[1][2] = intrinsic_matrix[1][2] * 4 / 9 - 4
        extrinsic_matrix = x["pose"]
        complete_3D_oracle = x["point_cloud"]
        complete_3D_oracle_original = complete_3D_oracle.copy()
        # k_ambiguity = -1
        one_frame_pc = np.array([])
        if args.not_entire:
            density = args.density
            k = args.k
            k_first = True
            complete_3D_oracle = down_sampling(np.asnumpy(complete_3D_oracle), density, k, k_first)

            project_positions_flat, project_positions_float, project_positions, _points_start, colors = projection_oracle(
                np.asarray(x["one_frame_point_cloud"]), extrinsic_matrix, intrinsic_matrix, k_ambiguity=-1,
                round=True)  # k_ambiguity)

            # print(project_positions_flat.shape)
            project_positions_float[:, :2] = project_positions_flat[:, :2].astype(np.float16) + 0.5
            # print(project_positions_float, project_positions_float.shape)
            project_positions_float = np.hstack(
                [project_positions_float, np.ones((project_positions_float.shape[0], 1))])
            # print("project_positions_flat", project_positions_float)
            # print("before project_positions", project_positions)
            project_positions = project_positions_float * project_positions[:, 2:3]
            # print("after project_positions", project_positions)
            project_positions = project_positions.T
            # print(intrinsic_matrix.shape, project_positions.shape)
            points_camera_cord = np.linalg.inv(intrinsic_matrix) @ project_positions
            positions = extrinsic_matrix @ np.vstack([points_camera_cord, np.ones((1, points_camera_cord.shape[1]))])
            positions = positions.T
            one_frame_pc = np.hstack([positions[:, :3], colors])

            project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
                one_frame_pc, extrinsic_matrix, intrinsic_matrix, k_ambiguity=-1, round=False)  # k_ambiguity)
            # print("start find 2d image")
            image, second_image, second_pixel_closest_point, pixel_points, _ = find_2d_image(project_positions_flat,
                                                                                          project_positions,
                                                                                          points_start,
                                                                                          colors, intrinsic_matrix,
                                                                                          need_second_img=True)
            # image_folder = f"./k_ambiguity_{k_ambiguity}/test_round/"
            # if not os.path.exists(image_folder):
            #     os.makedirs(image_folder)
            # matplotlib.image.imsave(f"{image_folder}/{i}.png", image.get())
            if complete_3D_oracle != None:
                print(complete_3D_oracle.shape, x["one_frame_point_cloud"].shape)
                complete_3D_oracle = np.vstack((complete_3D_oracle, one_frame_pc))
            else:
                print("Only single pc")
                complete_3D_oracle = one_frame_pc

        # image_folder = f"./k{k}_density{density}/test/"
        # if not os.path.exists(image_folder):
        #     os.makedirs(image_folder)
        # matplotlib.image.imsave(f"{image_folder}/{i}.png", image.get())
        '''
        outlier_rate = find_outlier_rate(image, second_image, args.L1_threshold)
        print("outlier_rate", outlier_rate)
        k_max, overflow_rate = find_k(project_positions_float, image, second_image, args.L1_threshold, second_pixel_closest_point, args.K)
        # s.quantile(0.95)
        print("k_max", k_max, overflow_rate) # 0.04041666666666666 K=100
        '''
        # k_max = args.K

        axis_range = [-args.partial, args.partial]
        axis = args.transtype[-2:]
        image_list = []
        if args.save_k_samples < 0:
            h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])
            image_shape = [h, w]
            if args.debug:
                table_sup_inf = numpy.load('table_sup_inf_1000.npy')
            else:
                table_sup_inf = find_table_sup_inf(complete_3D_oracle, extrinsic_matrix, intrinsic_matrix, axis, axis_range, image_shape)
                # np.save("table_sup_inf_100.npy", table_sup_inf)
            if args.exact:
                _diff = table_sup_inf[:, :, :, 0] - table_sup_inf[:, :, :, 1]
                diff_flat = numpy.asarray(_diff.flatten())
                # print(diff_flat.shape)
                diff_flat = numpy.delete(diff_flat, numpy.where(diff_flat >= 100))
                # print(diff_flat.shape)
                # diff_flat = numpy.delete(diff_flat, numpy.where(diff_flat < 1.1*(axis_range[1] - axis_range[0]) / RESOLUTION))
                # print(diff_flat.shape)
                Delta = numpy.min(diff_flat.flatten())
                print(f"exact Delta: {Delta}")
                num_slice_ = int(np.floor((axis_range[1] - axis_range[0]) / Delta) + 1)
                print(f"exact num_slice: {num_slice_}")
            else:
                project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
                    complete_3D_oracle, extrinsic_matrix, intrinsic_matrix, k_ambiguity=-1, no_filter_frustum=True)  # k_ambiguity)
                # print("start find 2d image")
                image, second_image, second_pixel_closest_point, pixel_points, _ = find_2d_image(project_positions_flat,
                                                                                                 project_positions,
                                                                                                 points_start,
                                                                                                 colors,
                                                                                                 intrinsic_matrix,
                                                                                                 need_second_img=True, no_filter_frustum=True)
                delta = 0
                if args.not_entire:
                    delta = calculate_delta(complete_3D_oracle_original, one_frame_pc,  extrinsic_matrix, intrinsic_matrix, axis, axis_range, image.shape)
                Delta = find_Delta(axis, axis_range, intrinsic_matrix, points_start, image.shape, pixel_points, table_sup_inf, delta,
                                   entire=not args.not_entire, k_ambiguity= -1) #k_ambiguity
            print(f"Delta: {Delta}")
            num_slice = int(np.floor((axis_range[1] - axis_range[0]) / Delta) + 1)
            print(f"num_slice: {num_slice}")
        else:
            num_slice = args.save_k_samples+1
            num_slice_ = num_slice
            Delta = (axis_range[1] - axis_range[0]) / num_slice
        print(f"num_slice: {num_slice}", i)
        if args.save_k_samples < 0:
            # \t{num_slice_}
            print(f'num: {i}\tDelta: {Delta}\tNum Slice: {num_slice}\tTime: {str(datetime.timedelta(seconds=(time() - global_before_time)))}', file=f,
                  flush=True)
            continue
        for slide_i in trange(num_slice):
            alpha_i = axis_range[0] + slide_i * (axis_range[1] - axis_range[0]) / (
                        np.floor((axis_range[1] - axis_range[0]) / Delta) + 1)
            extrinsic_matrix_i = find_new_extrinsic_matrix(extrinsic_matrix, axis, alpha_i)
            # print(alpha_i, Delta, axis_range)
            # project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
            #     complete_3D_oracle, extrinsic_matrix_i, intrinsic_matrix, k_ambiguity=k_ambiguity)
            project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
                complete_3D_oracle, extrinsic_matrix_i, intrinsic_matrix, k_ambiguity=-1)  # k_ambiguity)
            image_i, _, _, _, _ = find_2d_image(project_positions_flat, project_positions, points_start, colors,
                                             intrinsic_matrix, need_second_img=False)
            if args.save_k_samples > 0:
                image_folder = f"{args.aliasfile}_imgs_fixed_mew/" + '%03d' % i + "/"
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                matplotlib.image.imsave(f"{image_folder}/" + '%05d' % slide_i + ".png", image_i.get())
            image_list.append(image_i.reshape(-1))
        image_all = np.stack(image_list)
        '''
        # print("image_all[:-1, :] - image_all[1:, :]", (image_all[:-1, :] - image_all[1:, :]).shape)
        # print("np.sum(np.square(image_all[:-1, :] - image_all[1:, :]),axis=1)", np.sum(np.square(image_all[:-1, :] - image_all[1:, :]),axis=1).shape)
        max_next_neighbor_image_all = find_next_neighbor_square(image_all) #np.square(image_all[:-1, :] - image_all[1:, :])3
        max_last_neighbor_image_all = find_last_neighbor_square(image_all)
        # print(max_neighbor_image_all.reshape(max_neighbor_image_all.shape[0], -1).shape)
        # print(np.sum(max_neighbor_image_all.reshape(max_neighbor_image_all.shape[0], -1), axis=1))
        print("next", np.sqrt(np.sum(max_next_neighbor_image_all.reshape(max_next_neighbor_image_all.shape[0], -1), axis=1)))
        print("last", np.sqrt(np.sum(max_last_neighbor_image_all.reshape(max_last_neighbor_image_all.shape[0], -1), axis=1)))

        '''
        # max_next_neighbor_image_all = np.square(image_all.reshape(image_all.shape[0], -1)[:-1, :] - image_all.reshape(image_all.shape[0], -1)[1:, :])
        # max_last_neighbor_image_all = np.square(image_all.reshape(image_all.shape[0], -1)[:-1, :] - image_all.reshape(image_all.shape[0], -1)[1:, :])
        #
        #
        # M = np.amax(np.sqrt(np.minimum(np.sum(max_next_neighbor_image_all.reshape(max_next_neighbor_image_all.shape[0], -1), axis=1),
        #                                np.sum(max_last_neighbor_image_all.reshape(max_last_neighbor_image_all.shape[0], -1), axis=1))/2), axis=0)
        # neighbor_all = np.square(image_all.reshape(image_all.shape[0], -1)[:-1, :] - image_all.reshape(image_all.shape[0], -1)[1:, :])
        # M = np.amax(np.sqrt(np.sum(neighbor_all.reshape(neighbor_all.shape[0], -1), axis=1) / 2), axis=0)
        M = np.amax(np.sqrt(np.sum(np.square(image_all[:-1, :] - image_all[1:, :]), axis=1) / 2), axis=0)
        print("M", M)
        # image_folder = "./projected_images/test/"
        # if not os.path.exists(image_folder):
        #     os.makedirs(image_folder)
        # # plt.axis('off')
        # # plt.savefig(f"{image_folder}/{object_name}_{folder}_{index}.png", dpi=1)
        # matplotlib.image.imsave(f"{image_folder}/{i}.png", image_i)
        # matplotlib.image.imsave(f"test.png", image.get())

        #
        # global_max_aliasing = 0.0
        # d = 360.0 / (args.slice * args.subslice)
        # for j in range(args.slice):
        #     max_aliasing = 0.0
        #
        #     base_ang = 360.0 * j / args.slice
        #
        #     if 360.0 * j / args.slice > args.partial and 360.0 - 360.0 * (j + 1) / args.slice > args.partial:
        #         continue
        #
        #     base_img = rotationT.rotation_adder.proc(x, base_ang)
        #     # exit(0)
        #     L = get_finer_lipschitz_bound(x, rotationT.rotation_adder.mask, base_ang, base_ang + 360.0 / args.slice)
        #
        #     ang_l = base_ang
        #     x_k_l = rotationT.rotation_adder.proc(x, ang_l)
        #     alias_k_l = torch.sum((x_k_l - base_img) * (x_k_l - base_img))
        #
        #     for k in range(args.subslice):
        #         ang_r = base_ang + (k+1) * d
        #         x_k_r = rotationT.rotation_adder.proc(x, ang_r)
        #         alias_k_r = torch.sum((x_k_r - base_img) * (x_k_r - base_img))
        #
        #         now_max_alias = (alias_k_l + alias_k_r) / 2.0 + (d * L) / 2.0
        #         max_aliasing = max(now_max_alias.item(), max_aliasing)
        #
        #         ang_l = ang_r
        #         x_k_l = x_k_r
        #         alias_k_l = alias_k_r
        #
        #     global_max_aliasing = max(global_max_aliasing, max_aliasing)
        #     if j % args.verbstep == 0:
        #         print(i, f'{j}/{args.slice}', max_aliasing, global_max_aliasing, str(datetime.timedelta(seconds=(time() - before_time))))
        #         before_time = time()

        print(f'{i}\t{M}\t{num_slice}\t{str(datetime.timedelta(seconds=(time() - global_before_time)))}', file=f,
              flush=True)
    # del complete_3D_oracle, project_positions_flat, project_positions_float, project_positions, points_start, colors,  neighbor_all
    # np._default_memory_pool.free_all_blocks()
    f.close()

    # # debug: compare manual rotation and library rotation
    # # now they are totally equal, means the library rotation is truly bi-linear
    # for i in range(10):
    #     img = dataset[i][0]
    #     angle = rotationT.rotation_adder.gen_param()
    #     my_out = rotate(img, angle, rotationT.rotation_adder.mask)
    #     my_out = rotationT.rotation_adder.masking(my_out)
    #     lib_out = rotationT.rotation_adder.proc(img, angle)
    #     print(i, angle, torch.sum((my_out - lib_out) * (my_out - lib_out)))
    #     visualize(my_out, f'test/test/transform/{args.dataset}/{args.split}/manual-rotation/{i}-{int(angle)}-man.bmp')
    #     visualize(lib_out, f'test/test/transform/{args.dataset}/{args.split}/manual-rotation/{i}-{int(angle)}-lib.bmp')


