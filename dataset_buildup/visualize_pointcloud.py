import matplotlib.pyplot as plt
import numpy as np
# import cupy as cp
import open3d as o3d

def main():
    # load point cloud
    ''' 
    path = "dataset/NONE/point_cloud.npy"
    point_cloud = np.load(path)
    # point_cloud = cp.asarray(point_cloud)
    # point_num = point_cloud.shape[0]

    # down sampling
    # point_index = np.arange(0, point_num, 20)

    point_cloud_o3d = o3d.t.geometry.PointCloud()#.cuda()
    point_cloud_o3d.point["positions"] = o3d.core.Tensor(point_cloud[:, 0: 3], o3d.core.float32)#.cuda()
    point_cloud_o3d.point["colors"] = o3d.core.Tensor(point_cloud[:, 3: 6], o3d.core.float32)#.cuda()


    point_cloud_o3d = point_cloud_o3d.voxel_down_sample(0.0025)
    # point_cloud_o3d = o3d.geometry.PointCloud.voxel_down_sample(point_cloud_o3d, 0.01)
    original_positions = np.asarray((point_cloud_o3d.cpu().point["positions"]).numpy())
    colors = np.asarray((point_cloud_o3d.cpu().point["colors"]).numpy())
    p = np.hstack((original_positions, colors))

    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector((point_cloud_o3d.cpu().point["positions"]).numpy())
    pt.colors = o3d.utility.Vector3dVector((point_cloud_o3d.cpu().point["colors"]).numpy())
    '''
    pt = o3d.io.read_point_cloud("/home/hanjiang/projects/Kinova-Gen3-Webots/webots/controllers/scan_controller/projected_imgs/telephone/test_r10_y-180_pc.pcd")
    # pt = pt.voxel_down_sample(0.0025)
    # # pt += o3d.t.io.read_point_cloud()
    # pt += o3d.io.read_point_cloud("background.pcd")
    # pt = pt.voxel_down_sample(0.0025)
    # print("add background")
    # pt += o3d.io.read_point_cloud("floor.pcd")
    # pt = pt.voxel_down_sample(0.0025)
    # print("add floor")
    # pt += o3d.io.read_point_cloud("ceiling.pcd")
    # pt = pt.voxel_down_sample(0.0025)
    # print("add ceiling")
    # pt_ground.points = o3d.utility.Vector3dVector((point_cloud_o3d.cpu().point["positions"]).numpy())
    # pt_ground.colors = o3d.utility.Vector3dVector((point_cloud_o3d.cpu().point["colors"]).numpy())

    o3d.visualization.draw_geometries([pt])
    # o3d.io.write_point_cloud("whole_pc_0.0025.pcd", pt)
    # point_cloud_np = np.hstack([np.asarray(pt.points), np.asarray(pt.colors)])
    # print(point_cloud_np.shape)
    # np.save("whole_pc_0.0025.npy", point_cloud_np)

    '''
    {
        "class_name" : "ViewTrajectory",
        "interval" : 29,
        "is_loop" : false,
        "trajectory" : 
        [
            {
                "boundingbox_max" : [ 1.2699999809265137, 1.25, 1.2999999523162842 ],
                "boundingbox_min" : [ -1.2200000286102295, -1.309999942779541, 0.0 ],
                "field_of_view" : 60.0,
                "front" : [ -0.59434945072804357, -0.70663491097533926, 0.38394769566979753 ],
                "lookat" : [ 0.02499997615814209, -0.029999971389770508, 0.64999997615814209 ],
                "up" : [ 0.28648938729188922, 0.26006113763439542, 0.92211281070236839 ],
                "zoom" : 0.69999999999999996
            }
        ],
        "version_major" : 1,
        "version_minor" : 0
    }
    '''


    # p = point_cloud[point_index]

    # visualize
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # max_length = np.abs(p[:, 0: 3]).max()
    # ax.set_xlim3d([-max_length, max_length])
    # ax.set_ylim3d([-max_length, max_length])
    # ax.set_zlim3d([-max_length, max_length])
    # ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=p[:, 3: 6])
    # ax.set_axis_off()
    # plt.show()


if __name__ == '__main__':
    main()
