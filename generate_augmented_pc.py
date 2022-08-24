import numpy as np
import open3d as o3d
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import open3d as o3d
import laspy
from sklearn.cluster import DBSCAN


def generate_augmented_tiles(tilecodes, parameters, translation=False, scaling=False, cloud_rotation=False, 
                            add_normals=True, manual_check=True):
    class_names = {4: 'street_light', 5: 'street_sign', 6: 'traffic_light', 8: 'city_bench', 9: 'rubbish_bin'}
    class_label = parameters[0] # label of object of interest to crop tile for.
    class_name = class_names[class_label]
    tile_radius = parameters[1] # radius of tile.
    translation_factor_limits = parameters[2]
    scaling_factor_limits = parameters[3] # world scaling factor to scale complete tile.
    degree_limit_tile_rotation = parameters[4] # maximum rotation degree of complete tile.
    reproduce_percentage = parameters[5] # what probability to make new tile out of object of interest. range 0-1.
    augm_type = parameters[6]
    total = 0

    for tilecode in tilecodes:

        # load default point cloud
        in_file = 'dataset/point_clouds/' + tilecode + '.laz'
        laz_file = laspy.read(in_file)
        all_xyz = (np.vstack((laz_file.x, laz_file.y, laz_file.z)).T.astype(np.float32))
        all_x, all_y, all_z = all_xyz[:,0], all_xyz[:,1], all_xyz[:,2]
        all_r, all_g, all_b = np.asarray(laz_file.red), np.asarray(laz_file.green), np.asarray(laz_file.blue)
        all_intensity = np.asarray(laz_file.intensity)
        all_labels = laz_file.label

        # compute point normals if not in original tile
        if add_normals:
            object_pcd = o3d.geometry.PointCloud()
            points = np.stack((all_x, all_y, all_z), axis=-1)
            object_pcd.points = o3d.utility.Vector3dVector(points)
            object_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            normals = np.matrix.round(np.array(object_pcd.normals), 2)
            normals_of_interest = np.squeeze(np.where(normals_z <= -0.95))
            mean_z = np.mean(all_z[normals_of_interest])
            normal_indxs_to_change = ((all_z[normals_of_interest] > (mean_z - 2)) & (all_z[normals_of_interest] < (mean_z + 2)))
            normals_z[normals_of_interest[normal_indxs_to_change]] = np.absolute(normals_z[normals_of_interest[normal_indxs_to_change]])
            normals[:,2] = normals_z
            all_normals_x, all_normals_y, all_normals_z = normals[:,0], normals[:,1], normals[:,2]
        
        # crop tile around object of interest and apply augmentation methods
        pnt_idxs = np.squeeze(np.where(all_labels == class_label))
        coordinates = all_xyz[pnt_idxs]
        clustering = DBSCAN().fit(coordinates)
        print('tilecode: {}'.format(tilecode))
        print('unique clusters with label {}: {}'.format(class_label, len(np.unique(clustering.labels_))))
        if len(np.unique(clustering.labels_)) > 0:

            for item_id, cluster_label in enumerate(np.unique(clustering.labels_)):
                if random.uniform(0, 1) < reproduce_percentage:
                    next_tile = False
                     
                    # Precalculate cropped tile dimensions
                    cluster_point_idxs = np.where(clustering.labels_ == cluster_label)
                    cluster_points = coordinates[cluster_point_idxs]
                    mean_x, mean_y = round(np.mean(cluster_points[:,0]),2), round(np.mean(cluster_points[:,1]),2)
                    
                    min_x_cropped_tile = max(mean_x - tile_radius, min(np.asarray(laz_file.x)))
                    max_x_cropped_tile = min(mean_x + tile_radius, max(np.asarray(laz_file.x)))
                    min_y_cropped_tile = max(mean_y - tile_radius, min(np.asarray(laz_file.y)))
                    max_y_cropped_tile = min(mean_y + tile_radius, max(np.asarray(laz_file.y)))
                    condition = ((all_x>=min_x_cropped_tile) & (all_x<=max_x_cropped_tile) & (all_y>=min_y_cropped_tile) & (all_y<=max_y_cropped_tile))
                    
                    mean_z = round(np.mean(cluster_points[:,2]),2)
                    min_z_cropped_tile = max(mean_z - tile_radius, min(np.asarray(laz_file.z)))
                    max_z_cropped_tile = min(mean_z + tile_radius, max(np.asarray(laz_file.z)))
                    condition = ((all_x>=min_x_cropped_tile) & (all_x<=max_x_cropped_tile) & 
                                    (all_y>=min_y_cropped_tile) & (all_y<=max_y_cropped_tile) &
                                    (all_z>=min_z_cropped_tile) & (all_z<=max_z_cropped_tile))
                    
                    condition_indxs = np.asarray(condition).nonzero()[0]

                    # Crop tile
                    xyz_cropped = all_xyz[condition_indxs]
                    r_cropped, g_cropped, b_cropped = all_r[condition_indxs], all_g[condition_indxs], all_b[condition_indxs]
                    intensity_cropped = all_intensity[condition_indxs]
                    labels_cropped = all_labels[condition_indxs]
                    if add_normals:
                        normals_x_cropped = all_normals_x[condition_indxs]
                        normals_y_cropped = all_normals_y[condition_indxs]
                        normals_z_cropped = all_normals_z[condition_indxs]
                    _, cluster_point_idxs_cropped, _ = np.intersect1d(list(condition_indxs), pnt_idxs[np.squeeze(cluster_point_idxs)], return_indices=True)

                    while next_tile == False:
                        xyz = xyz_cropped
                        r, g, b = r_cropped, g_cropped, b_cropped
                        intensity = intensity_cropped
                        labels = labels_cropped
                        if add_normals:
                            normals_x, normals_y, normals_z = normals_x_cropped, normals_y_cropped, normals_z_cropped

                        # translate complete tile with predefined translation factor
                        if translation:
                            xyx, x_value, y_value = translate_point_cloud(xyz, translation_factor_limits)
                            value = str(x_value) + '_' + str(y_value)
                        
                        # scale complete tile with predefined scaling factor
                        if scaling:
                            xyz, value = scale_point_cloud(xyz, scaling_factor_limits)

                        # rotate complete tile with predefined degree limit
                        if cloud_rotation:
                            xyz, value = rotate_point_cloud(xyz, degree_limit_tile_rotation)

                        # Generate and save tile
                        if manual_check:
                            object_pcd = o3d.geometry.PointCloud()
                            points = np.stack((xyz[:,0], xyz[:,2], xyz[:,1]), axis=-1)
                            object_pcd.points = o3d.utility.Vector3dVector(points)
                            colors = np.dstack((r, g, b))[0]/255
                            object_pcd.colors = o3d.utility.Vector3dVector(colors)
                            o3d.visualization.draw_geometries([object_pcd])
                            save = input('save [y/n]: ')
                        else:
                            save = 'y'

                        if save == 'y':
                            next_tile = True
                            cropped_laz_file = laspy.create(file_version="1.2", point_format=3)
                            cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                                name="label", type="uint8", description="Labels"))
                            cropped_laz_file.x, cropped_laz_file.y, cropped_laz_file.z = xyz[:,0], xyz[:,1], xyz[:,2]
                            cropped_laz_file.red, cropped_laz_file.green, cropped_laz_file.blue = r, g, b
                            cropped_laz_file.intensity = intensity
                            cropped_laz_file.label = labels

                            # add normals
                            if add_normals:
                                cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                                name="normal_x", type="float", description="normal_x"))
                                cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                                name="normal_y", type="float", description="normal_y"))
                                cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                                name="normal_z", type="float", description="normal_z"))
                                cropped_laz_file.normal_x = normals_x
                                cropped_laz_file.normal_y = normals_y
                                cropped_laz_file.normal_z = normals_z

                            # save tile with information in file name
                            cropped_laz_file.write("dataset/tiles/{}/{}_{}_{}_{}_{}.laz".format(augm_type, tilecode, augm_type, class_name, item_id, tile_radius))

                        elif save == 'n':
                            next_tile = True

def translate_point_cloud(coordinates, translation_factor_limits):
    translation_factor_x = round(random.uniform(translation_factor_limits[0], translation_factor_limits[1]), 1)
    translation_factor_y = round(random.uniform(translation_factor_limits[0], translation_factor_limits[1]), 1)
    print('translation x: {}, translation y: {}'.format(translation_factor_x, translation_factor_y))
    return coordinates + [translation_factor_x, translation_factor_y, 0], translation_factor_x, translation_factor_y

def scale_point_cloud(coordinates, scaling_factor_limits):
    mean_x, mean_y = round(np.mean(coordinates[:,0]),2), round(np.mean(coordinates[:,1]),2)
    scaling_factor = 1
    while scaling_factor == 1:
        scaling_factor = np.round(random.uniform(scaling_factor_limits[0], scaling_factor_limits[1]), 2)
    print('scaling factor: {}'.format(scaling_factor))
    translated_coordinates = coordinates - [mean_x, mean_y, 0]
    scaled_coordinates = translated_coordinates * [scaling_factor, scaling_factor, scaling_factor]
    return scaled_coordinates + [mean_x, mean_y, 0], scaling_factor

def rotate_point_cloud(coordinates, degree_limit_tile_rotation):
    degrees = random.randint(0, degree_limit_tile_rotation)
    print('point cloud rotation degrees: {}'.format(degrees))
    phi = degrees*np.pi/180
    rotation_matrix =   [[math.cos(phi),-math.sin(phi), 0],
                        [math.sin(phi), math.cos(phi), 0],
                        [0, 0, 1]]
    mean_x, mean_y = round(np.mean(coordinates[:,0]),2), round(np.mean(coordinates[:,1]),2)
    translated_coordinates = coordinates - [mean_x, mean_y, 0]
    rotated_coordinates = (translated_coordinates @ rotation_matrix) + [mean_x, mean_y, 0]
    return rotated_coordinates, degrees

def rotate_object(coordinates, cluster_point_idxs, degree_limit_object_rotation):
    degrees = random.randint(-degree_limit_object_rotation,degree_limit_object_rotation)
    print('object rotation degrees: {}'.format(degrees))
    phi = degrees*np.pi/180
    rotation_matrix =   [[math.cos(phi),-math.sin(phi), 0],
                        [math.sin(phi), math.cos(phi), 0],
                        [0, 0, 1]]
    cluster_points = coordinates[cluster_point_idxs]
    mean_x, mean_y = round(np.mean(cluster_points[:,0]),2), round(np.mean(cluster_points[:,1]),2)
    translated_cluster_points = cluster_points - [mean_x, mean_y, 0]
    transformed_cluster_points = (translated_cluster_points @ rotation_matrix) + [mean_x, mean_y, 0]
    coordinates[np.squeeze(cluster_point_idxs)] = transformed_cluster_points
    return coordinates, degrees

class_names = {4: 'street_light', 5: 'street_sign', 6: 'traffic_light', 8: 'city_bench', 9: 'rubbish_bin'}
class_label = 9 # label of object of interest to crop tile for.
tilecodes = []
tile_radius = 10 # radius of tile for MOBA. set to 10**6 if MOBA should not be used.
translation_factor_limits = [-20,20] # maximum translation in x and y direction.
scaling_factor_limits = [0.85, 1.15] # world scaling factor to scale complete tile.
degree_limit_tile_rotation = 360 # maximum rotation degree of complete tile.
degree_limit_object_rotation = 180 # maximum rotation degree of object of interest.
augm_type = '' # augmentation type. should be set as ROTATION, TRANSLATION, SCALING or MOBA, depending on configuration.
reproduce_percentage = 1 # probability to make new tile out of object of interest. range 0-1.
parameters = [class_label, tile_radius, translation_factor_limits, scaling_factor_limits, degree_limit_tile_rotation,
                reproduce_percentage, augm_type]

generate_augmented_tiles(tilecodes, parameters, translation=False, scaling=False, 
                        cloud_rotation=False, add_normals=False, manual_check=True)

    