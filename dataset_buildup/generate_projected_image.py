import cupy as np
import numpy, time
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator, LinearNDInterpolator
from cupy import linalg as LA
import open3d as o3d
import pandas as pd
import glob, os
from tqdm import trange
import argparse


def check_consistency(rgb1, rgb2, threshold):
    if (rgb1[0] == 1. and rgb1[1] == 1. and rgb1[2] == 1.) or (rgb2[0] == 1. and rgb2[1] == 1. and rgb2[2] == 1.):
        return False, True
    return True, LA.norm(rgb2 - rgb1, ord=1) < threshold
 

def filter_frustum(x_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start,
                   colors):
    project_positions_flat = project_positions_flat[x_ge_0_index]
    project_positions_float = project_positions_float[x_ge_0_index]
    project_positions = project_positions[x_ge_0_index]
    points_start = points_start[x_ge_0_index]
    colors = colors[x_ge_0_index]
    # print(points_start.shape)
    return project_positions_flat, project_positions_float, project_positions, points_start, colors


def projection_oracle(point_cloud_npy, extrinsic_matrix, intrinsic_matrix):
    # load point cloud
    # point_cloud_npy = np.load(f"dataset_old/{object_name}/point_cloud.npy")

    # point_cloud = o3d.io.read_point_cloud(f"dataset/{object_name}/point_cloud.npy", format='xyzrgb', remove_nan_points=True, remove_infinite_points=True, print_progress=False)
    # print(point_cloud)
    #
    # point_cloud.points = open3d.utility.Vector3dVector(point_cloud)
    print(point_cloud_npy.shape)
    point_cloud = point_cloud_npy
    original_positions = point_cloud[:, 0: 3].astype(np.float16)
    colors = point_cloud[:, 3: 6].astype(np.float16)
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
    project_positions_flat = np.floor(project_positions_float).astype(np.short)

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

    return project_positions_flat, project_positions_float, project_positions, points_start, colors


def find_2d_image(project_positions_flat, project_positions, points_start, colors, intrinsic_matrix):
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
    # unique = np.unique(project_positions_flat)
    filtered_list = []
    # print(np.where(project_positions_flat == unique))
    print("project_positions_flat", project_positions_flat.shape)
    print("unique", unique.shape)
    print("colored_positions", colored_positions.shape)
    # print(np.max(project_positions_flat),np.max(unique))

    unique = unique.astype(np.short)
    unique_ = np.repeat(unique[np.newaxis, :], project_positions_flat.shape[0], axis=0)
    project_positions_flat_ = np.repeat(project_positions_flat[:, np.newaxis, :], unique.shape[0], axis=1)
    colored_positions_ = np.repeat(colored_positions[:, np.newaxis, :], unique.shape[0], axis=1)
    print("project_positions_flat", project_positions_flat_.dtype)
    print("unique", unique_.dtype)
    print("colored_positions", colored_positions_.dtype)
    same_positions_xy_index = (project_positions_flat_ == unique_)[:, :, 0] & (project_positions_flat_ == unique_)[:, :,
                                                                              1]
    depths_all = np.where(same_positions_xy_index, colored_positions_[:, :, 2], np.inf)

    filtered_positions = colored_positions_[np.argmin(depths_all, axis=0), np.arange(depths_all.shape[1])]

    # for unique_item in unique:
    #     # print("unique_item", unique_item)
    #     # print("######", project_positions_flat)
    #
    #     points_index_with_same_pixel_x = np.where(project_positions_flat[:, 0] == unique_item[0])[0]
    #     project_positions_flat_x = project_positions_flat[points_index_with_same_pixel_x]
    #     colored_positions_x = colored_positions[points_index_with_same_pixel_x]
    #     # print(unique_item[0], project_positions_flat_x, colored_positions_x)
    #
    #     points_index_with_same_pixel_y = np.where(project_positions_flat_x[:, 1] == unique_item[1])[0]
    #     project_positions_flat_xy = project_positions_flat_x[points_index_with_same_pixel_y]
    #     colored_positions_xy = colored_positions_x[points_index_with_same_pixel_y]
    #
    #     # print("$$$$$$$$$$$",unique_item, project_positions_flat[points_index_with_same_pixel][0])
    #     # print(b,unique_item,project_positions_flat.shape)
    #     # a = colored_positions[points_index_with_same_pixel] #numpy.unique(np.asnumpy(points_index_with_same_pixel), axis=0)
    #     # print("a", colored_positions_xy)
    #     depth = colored_positions_xy[:, 2]
    #     print("depth", depth.shape)
    #     min_index = np.argmin(depth)
    #     filtered_list.append(colored_positions_xy[min_index])
    #     assert 1==2
    #     # assert depth.shape[0] == colored_positions_xy.shape[0]
    #     # assert colored_positions.shape[0] == project_positions_flat.shape[0]
    # filtered_positions = np.array(filtered_list)
    # image = filtered_positions[:, 3:6]
    # print(filtered_positions[:, :3])
    image[filtered_positions[:, 1:2].astype("int").T[0], filtered_positions[:, :1].astype("int").T[0],
    :] = filtered_positions[:, 3:6]
    # print("project_positions_flat", project_positions_flat.shape)
    # print("project_positions", project_positions.shape)
    # print("colored_positions", colored_positions.shape)
    # print("unique", unique.shape)
    # print("filtered_positions", filtered_positions.shape)

    '''
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
            image[y, x] = np.asarray(colors[i])
        # elif dist < second_dists[y, x]:
        #     second_dists[y, x] = dist
        #     second_pixel_closest_point[y][x] = i
        #     second_image[y, x] = np.asarray(colors[i])
    '''
    return image, second_image, second_pixel_closest_point, pixel_points


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


def find_k(project_positions_float, image, second_image, L1_threshold, second_pixel_closest_point):
    # # calculate k
    h = image.shape[0]
    w = image.shape[1]
    k_max = 0
    k_max_rs_list = []
    for r in range(w):
        for s in range(h):
            for k in range(3, 1000):
                # cover_center = False
                # for pc in pixel_points[s][r]:
                valid, consistent = check_consistency(image[s][r], second_image[s][r], L1_threshold)
                if not valid or (valid and not consistent):
                    continue
                pc = second_pixel_closest_point[s][r]
                # print(pc, project_positions_float[pc])
                if k * r + 1 <= k * project_positions_float[pc][0] and k * project_positions_float[pc][
                    0] <= k * r + k - 1 \
                        and k * s + 1 <= k * project_positions_float[pc][1] and k * project_positions_float[pc][
                    1] <= k * s + k - 1:
                    # if check_consistency(colors[pc], colors[pixel_closest_point[s][r]], 50):
                    #     print(r, s, k, pc)
                    # cover_center=True
                    # if cover_center:
                    if k > k_max:
                        k_max = k
                    k_max_rs_list.append(k)
                    break
    # print(k_max)

    s = pd.Series(k_max_rs_list)
    return k_max, s


def check_exist_rho_inside_pixel(rho, axis_range, r, s):
    # print(rho[0](0), s, rho[1](0), r)
    assert np.floor(rho[0](0)) == r
    assert np.floor(rho[1](0)) == s
    # print(rho[0](range[0]), rho[0](range[1]), rho[1](range[0]), rho[1](range[1]), r, s)

    both_less_0_r = rho[0](axis_range[0]) - r < 0 and rho[0](axis_range[1]) - r < 0
    both_greater_0_r = rho[0](axis_range[0]) - r - 1 >= 0 and rho[0](axis_range[1]) - r - 1 >= 0
    both_less_0_s = rho[1](axis_range[0]) - s < 0 and rho[1](axis_range[1]) - s < 0
    both_greater_0_s = rho[1](axis_range[0]) - s - 1 >= 0 and rho[1](axis_range[1]) - s - 1 >= 0

    return not (both_less_0_s or both_greater_0_s or both_less_0_r or both_greater_0_r)


def find_delta_tz(intrinsic_matrix, tz_range, points_start, h, w, pixel_points, k=10):
    L_points = np.maximum(
        intrinsic_matrix[0][0] * np.abs(points_start[:, 0]) / ((points_start[:, 2] - tz_range[1]) ** 2),
        intrinsic_matrix[1][1] * np.abs(points_start[:, 1]) / ((points_start[:, 2] - tz_range[1]) ** 2))
    L_rs_list = [-1]
    # print(len(pixel_points), h, w)
    for r in range(w):
        for s in range(h):
            L_rs_pc_list = [-1]
            for pc_index in pixel_points[s][r]:
                rho = [
                    lambda tz: intrinsic_matrix[0][0] * points_start[pc_index, 0] / (points_start[pc_index, 2] - tz) +
                               intrinsic_matrix[0][2],
                    lambda tz: intrinsic_matrix[1][1] * points_start[pc_index, 1] / (points_start[pc_index, 2] - tz) +
                               intrinsic_matrix[1][2]]
                if check_exist_rho_inside_pixel(rho, tz_range, r, s):
                    L_rs_pc_list.append(L_points[pc_index])
            L_rs_list.append(max(L_rs_pc_list))
    # print(1 / (max(L_rs_list) * k), k)
    return 1 / (max(L_rs_list) * k)


def find_delta_tx(intrinsic_matrix, tx_range, points_start, h, w, pixel_points, k=10):
    L_points = intrinsic_matrix[0][0] / points_start[:, 2]
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
                    L_rs_pc_list.append(L_points[pc_index])
            L_rs_list.append(max(L_rs_pc_list))
    # print(1 / (max(L_rs_list) * k), k)
    return 1 / (max(L_rs_list) * k)


def find_delta_ty(intrinsic_matrix, ty_range, points_start, h, w, pixel_points, k=10):
    L_points = intrinsic_matrix[1][1] / points_start[:, 2]
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
                    L_rs_pc_list.append(L_points[pc_index])
            L_rs_list.append(max(L_rs_pc_list))
    # print(1 / (max(L_rs_list) * k), k)
    return 1 / (max(L_rs_list) * k)


def find_delta_rz(intrinsic_matrix, rz_range, points_start, h, w, pixel_points, k=10):
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
                    L_rs_pc_list.append(L_points[pc_index])
            L_rs_list.append(max(L_rs_pc_list))
    # print(1 / (max(L_rs_list) * k), k)
    return 1 / (max(L_rs_list) * k)


def find_delta_ry(intrinsic_matrix, ry_range, points_start, h, w, pixel_points, k=10):
    d_rho1 = np.where(points_start[:, 0] > 0,
                      intrinsic_matrix[0][0] * (points_start[:, 0] ** 2 + points_start[:, 2] ** 2) / ((points_start[:,
                                                                                                       0] * np.sin(
                          ry_range[0]) + points_start[:, 2] * np.cos(ry_range[0])) ** 2),
                      intrinsic_matrix[0][0] * (points_start[:, 0] ** 2 + points_start[:, 2] ** 2) / ((points_start[:,
                                                                                                       0] * np.sin(
                          ry_range[1]) + points_start[:, 2] * np.cos(ry_range[1])) ** 2))

    d_rho2 = np.where(points_start[:, 0] > 0, intrinsic_matrix[1][1] * np.abs(
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
                    L_rs_pc_list.append(L_points[pc_index])
            L_rs_list.append(max(L_rs_pc_list))
    # print(1 / (max(L_rs_list) * k), k)
    return 1 / (max(L_rs_list) * k)


def find_delta_rx(intrinsic_matrix, rx_range, points_start, h, w, pixel_points, k=10):
    d_rho1 = np.where(points_start[:, 1] < 0, intrinsic_matrix[0][0] * np.abs(
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

    d_rho2 = np.where(points_start[:, 1] < 0,
                      intrinsic_matrix[1][1] * (points_start[:, 1] ** 2 + points_start[:, 2] ** 2) / ((-points_start[:,
                                                                                                        1] * np.sin(
                          rx_range[0]) + points_start[:, 2] * np.cos(rx_range[0])) ** 2),
                      intrinsic_matrix[1][1] * (points_start[:, 1] ** 2 + points_start[:, 2] ** 2) / ((-points_start[:,
                                                                                                        1] * np.sin(
                          rx_range[1]) + points_start[:, 2] * np.cos(rx_range[1])) ** 2))

    # L_points = np.maximum(intrinsic_matrix[0][0]*np.sqrt(points_start[:, 0]**2 + points_start[:, 1]**2)/points_start[:, 2],
    #                       intrinsic_matrix[1][1]*np.sqrt(points_start[:, 0]**2 + points_start[:, 1]**2)/points_start[:, 2])

    L_points = np.maximum(d_rho1, d_rho2)
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
                    L_rs_pc_list.append(L_points[pc_index])
            L_rs_list.append(max(L_rs_pc_list))
    # print(1 / (max(L_rs_list) * k), k)
    return 1 / (max(L_rs_list) * k)


def find_delta(axis, axis_range, intrinsic_matrix, points_start, image_shape, pixel_points, k_max):
    h = image_shape[0]
    w = image_shape[1]
    if axis == 'tz':
        delta = find_delta_tz(intrinsic_matrix, axis_range, points_start, h, w, pixel_points, k=k_max)
        print('tz', np.mean(points_start, axis=0), axis_range, delta, (axis_range[1] - axis_range[0]) / delta)
    elif axis == 'tx':
        delta = find_delta_tx(intrinsic_matrix, axis_range, points_start, h, w, pixel_points, k=k_max)
        print('tx', np.mean(points_start, axis=0), axis_range, delta, (axis_range[1] - axis_range[0]) / delta)
    elif axis == 'ty':
        delta = find_delta_ty(intrinsic_matrix, axis_range, points_start, h, w, pixel_points, k=k_max)
        print('ty', np.mean(points_start, axis=0), axis_range, delta, (axis_range[1] - axis_range[0]) / delta)
    elif axis == 'rz':
        delta = find_delta_rz(intrinsic_matrix, axis_range, points_start, h, w, pixel_points, k=k_max)
        print('rz', np.mean(points_start, axis=0), axis_range, delta, (axis_range[1] - axis_range[0]) / delta)
    elif axis == 'ry':
        delta = find_delta_ry(intrinsic_matrix, axis_range, points_start, h, w, pixel_points, k=k_max)
        print('ry', np.mean(points_start, axis=0), axis_range, delta, (axis_range[1] - axis_range[0]) / delta)
    else:
        delta = find_delta_rx(intrinsic_matrix, axis_range, points_start, h, w, pixel_points, k=k_max)
        print('rx', np.mean(points_start, axis=0), axis_range, delta, (axis_range[1] - axis_range[0]) / delta)
    return delta


def find_new_extrinsic_matrix(extrinsic_matrix, axis, alpha):
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
        R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                      [np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]])
        t = np.array([[0, 0, 0]])
    elif axis == 'ry':
        R = np.array([[np.cos(alpha), 0, np.sin(alpha)],
                      [0, 1, 0],
                      [-np.sin(alpha), 0, np.cos(alpha)]])
        t = np.array([[0, 0, 0]])
    else:
        R = np.array([[1, 0, 0],
                      [0, np.cos(alpha), -np.sin(alpha)],
                      [0, np.sin(alpha), np.cos(alpha)]])
        t = np.array([[0, 0, 0]])
    rel_matrix = np.vstack((np.hstack((R, t.T)), np.array([0, 0, 0, 1])))
    return extrinsic_matrix @ rel_matrix


def down_sampling(point_cloud_npy, density, k=-1, k_first=True):
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


def find_pointcloud_oracle(whole_point_cloud, intrinsic_matrix, extrinsic_matrix, axis, axis_range):
    new_extrinsic_matrix_min = find_new_extrinsic_matrix(extrinsic_matrix, axis, axis_range[0])
    new_extrinsic_matrix_max = find_new_extrinsic_matrix(extrinsic_matrix, axis, axis_range[1])
    h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])
    pixel_np = np.array([[0, 0, 0.01],
                         [0, 0, 4],
                         [w, 0, 0.01],
                         [w, 0, 4],
                         [0, h, 0.01],
                         [0, h, 4],
                         [w, h, 0.01],
                         [w, h, 4]])
    min_vertex = new_extrinsic_matrix_min @ np.vstack(
        [np.linalg.inv(intrinsic_matrix) @ pixel_np.T, np.ones([1, pixel_np.shape[0]])])
    max_vertex = new_extrinsic_matrix_max @ np.vstack(
        [np.linalg.inv(intrinsic_matrix) @ pixel_np.T, np.ones([1, pixel_np.shape[0]])])
    bounding_vertex = np.hstack([min_vertex, max_vertex])[:3].T
    three_view_polygon = project_polygon(bounding_vertex)
    polygon_crop = o3d.visualization.SelectionPolygonVolume()
    polygon_crop.orthogonal_axis = 'Z'
    polygon_crop.bounding_polygon = o3d.utility.Vector3dVector(three_view_polygon[2])
    point_cloud_after_Z = polygon_crop.crop_point_cloud(whole_point_cloud)
    polygon_crop.orthogonal_axis = 'Y'
    polygon_crop.bounding_polygon = o3d.utility.Vector3dVector(three_view_polygon[1])
    point_cloud_after_ZY = polygon_crop.crop_point_cloud(point_cloud_after_Z)
    polygon_crop.orthogonal_axis = 'X'
    polygon_crop.bounding_polygon = o3d.utility.Vector3dVector(three_view_polygon[0])
    point_cloud_after_ZYX = polygon_crop.crop_point_cloud(point_cloud_after_ZY)


def project_image(complete_3D_oracle, intrinsic_matrix, extrinsic_matrix, axis, axis_range, resolution_ratio=8,
                  L1_threshold=0.2):
    # t1 = time.time()
    # print("start multiplying")
    project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
        complete_3D_oracle, extrinsic_matrix, intrinsic_matrix)
    t2 = time.time()
    # print("start find 2d image")
    image, second_image, second_pixel_closest_point, pixel_points = find_2d_image(project_positions_flat,
                                                                                  project_positions, points_start,
                                                                                  colors, intrinsic_matrix)
    # outlier_rate = find_outlier_rate(image, second_image, L1_threshold)
    # k_max, s = find_k(project_positions_float, image, second_image, L1_threshold, second_pixel_closest_point)
    # # s.quantile(0.95)
    # delta = find_delta(axis, axis_range, intrinsic_matrix, points_start, image.shape, pixel_points, k_max=s.quantile(0.95))
    # image_list = []
    # for i in trange(int(np.floor((axis_range[1] - axis_range[0]) / delta)+1)):
    #     alpha_i = axis_range[1] + i * (axis_range[1] - axis_range[0]) / (np.floor((axis_range[1] - axis_range[0]) / delta)+1)
    #     extrinsic_matrix_i = find_new_extrinsic_matrix(extrinsic_matrix, axis, alpha_i)
    #     project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
    #         complete_3D_oracle, extrinsic_matrix_i, intrinsic_matrix, density=density)
    #
    #     image_i, second_image, second_pixel_closest_point, pixel_points = find_2d_image(project_positions_flat,
    #                                                                                   project_positions, points_start,
    #                                                                                   colors, intrinsic_matrix)
    #     image_list.append(image_i)
    #     image_folder = "./projected_images/test/"
    #     if not os.path.exists(image_folder):
    #         os.makedirs(image_folder)
    #     # plt.axis('off')
    #     # plt.savefig(f"{image_folder}/{object_name}_{folder}_{index}.png", dpi=1)
    #     matplotlib.image.imsave(f"{image_folder}/{i}.png", image_i)
    # matplotlib.image.imsave(f"test.png", image.get())
    # t3 = time.time()
    # print("end", t3-t1, t3-t2, t2-t1)
    return image.get()

    # print(s.quantile(0.95))
    # plt.hist(k_max_rs_list, bins=40)
    # plt.show()
    # original density, original resolution, L1=0.2,k= 993

    # plt.imshow(image)
    # # plt.show()
    # image_folder= f"./projected_images/{object_name}/{folder}"
    # if not os.path.exists(image_folder):
    #     os.makedirs(image_folder)
    # # plt.axis('off')
    # # plt.savefig(f"{image_folder}/{object_name}_{folder}_{index}.png", dpi=1)
    # matplotlib.image.imsave(f"{image_folder}/{object_name}_{folder}_{index}.png", image)
    # # plt.imshow(second_image)
    # # plt.show()
    # with open(f'{object_name}_{folder}_results.txt', 'a') as f:
    #     f.write(str(index) + "\t" + str(k_max) + "\t" + str(s.quantile(0.95)) + "\t" + str(1-outlier_rate) + "\n")
    # return k_max, s.quantile(0.95), 1-outlier_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='which object to be projected')
    parser.add_argument("--index", type=int, default=19, help="which dataset")
    args = parser.parse_args()
    # pt = o3d.io.read_point_cloud("whole_pc_0.0025.pcd")
    # pt = np.load("dataset/NONE/point_cloud.npy")
    # pt = numpy.load("/home/hanjiang/projects/Kinova-Gen3-Webots/webots/controllers/scan_controller/dataset_0509/apple/train_pc.npy").astype("float16")
    # extrinsic_matrix = np.load("/home/hanjiang/projects/Kinova-Gen3-Webots/webots/controllers/scan_controller/dataset_0509/apple/train/pose_0.npy").astype("float64")
    # pt = numpy.load("/home/hanjiang/projects/Kinova-Gen3-Webots/webots/controllers/scan_controller/dataset_0509/apple/test_r6_y-180_pc.npy").astype("float64")
    # extrinsic_matrix = np.load("/home/hanjiang/projects/Kinaova-Gen3-Webots/webots/controllers/scan_controller/dataset_0509/apple/test/pose_r6_y-180_0.npy").astype("float64")
    axis = "tz"
    axis_range = [-0.1, 0.1]
    resolution_ratio = 8
    if_test = 1
    if if_test:

        # density = 0.001 # 340s
        # tx
        # k = 6
        # density = 0.01365
        # k_first = True

        # ty
        # k = 7
        # density = 0.0137
        # k_first = True

        # tz 8, 0.012
        k = 7
        density = 0.0133
        k_first = True

        # ry
        # k = 7
        # density = 0.0134
        # k_first = True

        # rx
        # k = 6
        # density = 0.01355
        # k_first = True

        # rz
        # k = 7
        # density = 0.0135
        # k_first = True

        raw_data_path = "../projected_large_reslt_imgs_tz"
        object_paths = sorted(glob.glob(raw_data_path + "/*"))
        for idx, object_path in enumerate(object_paths):
            # if idx % 20 != args.index: continue
            # if object_path != "../projected_large_reslt_imgs_ty/orange": continue
            print(object_path)
            # np.unique()
            projected_img_paths = f"../projected_test_{raw_data_path[-2:]}/test/{object_path.split('/')[-1]}"
            if not os.path.exists(projected_img_paths):
                os.makedirs(projected_img_paths)
            intrinsic_matrix = np.load(object_path + "/camera_intrinsic_matrix.npy")
            intrinsic_matrix[0][0] /= resolution_ratio
            intrinsic_matrix[1][1] /= resolution_ratio
            intrinsic_matrix[0][2] /= resolution_ratio
            intrinsic_matrix[1][2] /= resolution_ratio
            test_pose_paths = sorted(glob.glob(object_path + "/test/*.npy"))
            complete_3D_oracle_path_list = sorted(glob.glob(object_path + "/test_*_pc.npy"))
            oracle_dic = {}
            for oracle_path in complete_3D_oracle_path_list:
                pt = numpy.load(oracle_path)
                oracle = down_sampling(pt, density, k, k_first)
                np.save(oracle_path.replace("/test_", "/new_test_"), oracle)
                oracle_dic[oracle_path.split("/")[-1]] = oracle
            for i in trange(len(test_pose_paths)):
                test_pose_path = test_pose_paths[i]
                extrinsic_matrix = np.load(test_pose_path)
                complete_3D_oracle = oracle_dic["test_" + test_pose_path.split('/')[-1].split("_")[1] + "_" +
                                                test_pose_path.split('/')[-1].split("_")[2] + "_pc.npy"]
                image = project_image(np.asarray(complete_3D_oracle), intrinsic_matrix, extrinsic_matrix, axis,
                                      axis_range,
                                      resolution_ratio=resolution_ratio,
                                      L1_threshold=0.2)
                matplotlib.image.imsave(
                    projected_img_paths + '/' + test_pose_path.split('/')[-1].replace('pose', 'img').replace('npy',
                                                                                                             'png'),
                    image)
            #     break
            # break

        # load extrinsic matrix
        # extrinsic_matrix = np.load(f"dataset_old/{object_name}/{folder}/pose_{index}.npy")
        # print("start")
    else:
        # density = 0.001 # 340s
        density = 0.01  # 69s
        # density = 0.005 # 31s
        raw_data_path = "/home/hanjiang/projects/Kinova-Gen3-Webots/webots/controllers/scan_controller/dataset_new"
        # raw_data_path = '/media/hanjiang/Elements SE/robost_vision/dataset_bins'
        object_paths = sorted(glob.glob(raw_data_path + "/*"))
        for idx, object_path in enumerate(object_paths):
            if idx % 20 != args.index: continue
            print(object_path)
            # np.unique()
            projected_img_paths = f"../projected_dataset/train/{object_path.split('/')[-1]}"
            if not os.path.exists(projected_img_paths):
                os.makedirs(projected_img_paths)
            intrinsic_matrix = np.load(object_path + "/camera_intrinsic_matrix.npy")
            intrinsic_matrix[0][0] /= resolution_ratio
            intrinsic_matrix[1][1] /= resolution_ratio
            intrinsic_matrix[0][2] /= resolution_ratio
            intrinsic_matrix[1][2] /= resolution_ratio
            train_pose_paths = sorted(glob.glob(object_path + "/train/*.npy"))
            pt = numpy.load(object_path + "/train_pc.npy")
            complete_3D_oracle = down_sampling(pt, density)
            # complete_3D_oracle = pt
            for i in trange(len(train_pose_paths)):
                train_pose_path = train_pose_paths[i]
                extrinsic_matrix = np.load(train_pose_path)

                # pt = numpy.load("/home/hanjiang/projects/Kinova-Gen3-Webots/webots/controllers/scan_controller/dataset/telephone/test_r10_y-180_pc.npy").astype("float64")
                # extrinsic_matrix = np.load("/home/hanjiang/projects/Kinova-Gen3-Webots/webots/controllers/scan_controller/dataset/telephone/test/pose_r10_y-180_2.npy").astype("float64")

                # complete_3D_oracle = find_pointcloud_oracle(whole_point_cloud, intrinsic_matrix, extrinsic_matrix, axis, range)

                image = project_image(complete_3D_oracle, intrinsic_matrix, extrinsic_matrix, axis, axis_range,
                                      resolution_ratio=resolution_ratio,
                                      L1_threshold=0.2)
                matplotlib.image.imsave(
                    projected_img_paths + '/' + train_pose_path.split('/')[-1].replace('pose', 'img').replace('npy',
                                                                                                              'png'),
                    image)

    assert 1 == 3
    image_path_list = sorted(glob.glob("./dataset_old/telephone/train/*.png"))
    k_max = 0
    k_list = []
    i_k_max = -1

    k_95_max = 0
    k_95_list = []
    i_k_95_max = -1

    rate_max = 0
    rate_list = []
    i_rate_max = -1
    for i in trange(len(image_path_list)):
        k, k_95, consistent_rate = certification('telephone', i, "train")
        k_list.append(k)
        if k > k_max:
            k_max = k
            i_k_max = i
        k_95_list.append(k_95)
        if k_95 > k_95_max:
            k_95_max = k_95
            i_k_95_max = i
        rate_list.append(consistent_rate)
        if consistent_rate > rate_max:
            rate_max = consistent_rate
            i_rate_max = i
        if i > 10: break
    # s = pd.Series(k_95_list)
    # print(s.quantile(0.95))

    print(k_max, i_k_max)
    plt.hist(k_list, bins=200)
    plt.show()

    print(k_95_max, i_k_95_max)
    plt.hist(k_95_list, bins=200)
    plt.show()

    print(rate_max, i_rate_max)
    plt.hist(rate_list, bins=200)
    plt.show()
