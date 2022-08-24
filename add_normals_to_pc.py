import numpy as np
import open3d as o3d
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import open3d as o3d
import laspy
from sklearn.cluster import DBSCAN


def add_normals(tilecodes):
    
    for i, tilecode in enumerate(tilecodes):
        print(i, len(tilecodes))

        # load default point cloud
        in_file = 'dataset/point_clouds/' + tilecode + '.laz'
        laz_file = laspy.read(in_file)
        all_xyz = (np.vstack((laz_file.x, laz_file.y, laz_file.z)).T.astype(np.float32))
        all_x, all_y, all_z = all_xyz[:,0], all_xyz[:,1], all_xyz[:,2]
        r, g, b = np.asarray(laz_file.red), np.asarray(laz_file.green), np.asarray(laz_file.blue)
        intensity = np.asarray(laz_file.intensity)
        labels = laz_file.label

        # compute normals
        object_pcd = o3d.geometry.PointCloud()
        points = np.stack((all_x, all_y, all_z), axis=-1)
        object_pcd.points = o3d.utility.Vector3dVector(points)
        object_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.matrix.round(np.array(object_pcd.normals), 2)
        normals_z = normals[:,2]
        normals_of_interest = np.squeeze(np.where(normals_z <= -0.95))
        mean_z = np.mean(all_z[normals_of_interest])
        normal_indxs_to_change = ((all_z[normals_of_interest] > (mean_z - 2)) & (all_z[normals_of_interest] < (mean_z + 2)))
        normals_z[normals_of_interest[normal_indxs_to_change]] = np.absolute(normals_z[normals_of_interest[normal_indxs_to_change]])
        normals[:,2] = normals_z

        # generate new point cloud file, add data and save point cloud
        new_laz_file = laspy.create(file_version="1.2", point_format=3)
        new_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                            name="label", type="uint8", description="Labels"))
        new_laz_file.x, new_laz_file.y, new_laz_file.z = all_x, all_y, all_z
        new_laz_file.red, new_laz_file.green, new_laz_file.blue = r, g, b
        new_laz_file.intensity = intensity
        new_laz_file.label = labels
        new_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                        name="normal_x", type="float", description="normal_x"))
        new_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                        name="normal_y", type="float", description="normal_y"))
        new_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                        name="normal_z", type="float", description="normal_z"))
        new_laz_file.normal_x = normals[:,0]
        new_laz_file.normal_y = normals[:,1]
        new_laz_file.normal_z = normals[:,2]
        new_laz_file.write("dataset/point_clouds_with_normals/{}.laz".format(tilecode))