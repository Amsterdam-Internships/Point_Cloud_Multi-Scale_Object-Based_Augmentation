from os.path import join
import tensorflow as tf
import numpy as np
import time
import pickle
import argparse
import glob
import os
import sys

from core.models.RandLANet import Network as RandLANet
from core.models.SCFNet import Network as SCFNet
from core.models.CGANet import Network as CGANet
from core.models.GANet import Network as GANet


from core.tester import ModelTester
from calculate_clusters import CalculateClusters # ADDED
from core.helpers.helper_tool import DataProcessing as DP
import core.helpers.helper_filelists as utils

from configs.config_Amsterdam3D import *

from tensorflow.python.client import device_lib


class Amsterdam3D:
    def __init__(self, mode, in_folder, in_files):
        self.name = 'Amsterdam3D'

        # 10 Classes
        self.label_to_names = {0: 'road',
                               1: 'ground',
                               2: 'building',
                               3: 'tree',
                               4: 'street light',
                               5: 'street sign',
                               6: 'traffic light',
                               7: 'car',
                               8: 'city bench',
                               9: 'rubbish bin'}

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort(
            [k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])
        
        self.mode = mode

        if args.add_augmented_tiles:
            additional_information = args.augmentation_type + '_'
        else: additional_information = 'REGULAR_'

        # SPLIT 1
        if args.split == 1:
            additional_information += 'split_1'
            self.train_files = ['2621_9612', '2624_9604', '2685_9564', '2624_9603', '2624_9605', '2624_9607', '2628_9585', '2641_9588', '2617_9615', '2628_9601', '2633_9596', '2628_9599', '2615_9607', '2624_9608', '2615_9606', '2626_9581', '2628_9578', '2658_9538', '2628_9600', '2624_9617', '2643_9584', '2698_9613', '2605_9593', '2624_9610', '2632_9578', '2628_9598', '2628_9591', '2628_9590', '2624_9609', '2624_9597', '2619_9606', '2633_9598', '2620_9601', '2640_9590', '2624_9601', '2601_9615', '2632_9597', '2629_9580', '2633_9595', '2623_9621', '2640_9594', '2622_9606', '2632_9593', '2640_9598', '2660_9572', '2641_9587', '2628_9615', '2624_9618', '2640_9586', '2622_9589', '2617_9610', '2628_9577', '2624_9592', '2624_9616', '2629_9608', '2624_9615', '2614_9627', '2621_9590', '2624_9591', '2641_9584', '2621_9602', '2624_9593', '2640_9587', '2628_9604', '2624_9600', '2617_9614', '2628_9581', '2624_9602', '2641_9621', '2640_9584', '2629_9598', '2624_9613', '2624_9611', '2628_9576', '2641_9585', '2628_9579', '2628_9597', '2645_9582', '2628_9592', '2607_9589', '2628_9580', '2614_9600', '2614_9628', '2631_9608', '2635_9596', '2641_9611', '2614_9601', '2624_9588', '2641_9583', '2624_9590', '2631_9609', '2640_9607', '2628_9596', '2625_9598', '2641_9593', '2640_9583']
            self.val_files = ['2616_9619', '2616_9588', '2628_9602', '2640_9591', '2611_9602', '2641_9590', '2621_9588', '2622_9590', '2630_9603', '2631_9605', '2600_9621', '2615_9617', '2640_9595', '2640_9599', '2615_9610', '2660_9538', '2640_9620', '2629_9579', '2640_9606', '2624_9594', '2628_9586', '2628_9603', '2624_9612', '2626_9595']
            self.test_files = ['2624_9596', '2630_9602', '2628_9583', '2640_9610', '2628_9595', '2617_9606', '2622_9588', '2624_9599', '2622_9603', '2640_9611', '2628_9616', '2624_9595', '2624_9606', '2624_9589', '2611_9619', '2616_9606', '2628_9610', '2628_9582', '2628_9584', '2640_9588']

        # SPLIT 2
        elif args.split == 2:
            additional_information += 'split_2'
            self.train_files = ['2640_9586', '2622_9589', '2617_9610', '2628_9577', '2624_9592', '2624_9616', '2629_9608', '2624_9615', '2614_9627', '2621_9590', '2624_9591', '2641_9584', '2621_9602', '2624_9593', '2640_9587', '2628_9604', '2624_9600', '2617_9614', '2628_9581', '2624_9602', '2641_9621', '2640_9584', '2629_9598', '2624_9613', '2624_9611', '2628_9576', '2641_9585', '2628_9579', '2628_9597', '2645_9582', '2628_9592', '2607_9589', '2628_9580', '2614_9600', '2614_9628', '2631_9608', '2635_9596', '2641_9611', '2614_9601', '2624_9588', '2641_9583', '2624_9590', '2631_9609', '2640_9607', '2628_9596', '2625_9598', '2641_9593', '2640_9583', '2616_9619', '2616_9588', '2628_9602', '2640_9591', '2611_9602', '2641_9590', '2621_9588', '2622_9590', '2630_9603', '2631_9605', '2600_9621', '2615_9617', '2640_9595', '2640_9599', '2615_9610', '2660_9538', '2640_9620', '2629_9579', '2640_9606', '2624_9594', '2628_9586', '2628_9603', '2624_9612', '2626_9595', '2624_9596', '2630_9602', '2628_9583', '2640_9610', '2628_9595', '2617_9606', '2622_9588', '2624_9599', '2622_9603', '2640_9611', '2628_9616', '2624_9595', '2624_9606', '2624_9589', '2611_9619', '2616_9606', '2628_9610', '2628_9582', '2628_9584', '2640_9588']
            self.val_files = ['2621_9612', '2624_9604', '2685_9564', '2624_9603', '2624_9605', '2624_9607', '2628_9585', '2641_9588', '2617_9615', '2628_9601', '2633_9596', '2628_9599', '2615_9607', '2624_9608', '2615_9606', '2626_9581', '2628_9578', '2658_9538', '2628_9600', '2624_9617', '2643_9584', '2698_9613', '2605_9593', '2624_9610']
            self.test_files = ['2632_9578', '2628_9598', '2628_9591', '2628_9590', '2624_9609', '2624_9597', '2619_9606', '2633_9598', '2620_9601', '2640_9590', '2624_9601', '2601_9615', '2632_9597', '2629_9580', '2633_9595', '2623_9621', '2640_9594', '2622_9606', '2632_9593', '2640_9598', '2660_9572', '2641_9587', '2628_9615', '2624_9618']

        # SPLIT 3
        elif args.split == 3:
            additional_information += 'split_3'
            self.train_files = ['2621_9612', '2624_9604', '2685_9564', '2624_9603', '2624_9605', '2624_9607', '2628_9585', '2641_9588', '2617_9615', '2628_9601', '2633_9596', '2628_9599', '2615_9607', '2624_9608', '2615_9606', '2626_9581', '2628_9578', '2658_9538', '2628_9600', '2624_9617', '2643_9584', '2698_9613', '2605_9593', '2624_9610', '2632_9578', '2628_9598', '2628_9591', '2628_9590', '2624_9609', '2624_9597', '2619_9606', '2633_9598', '2620_9601', '2640_9590', '2624_9601', '2601_9615', '2632_9597', '2629_9580', '2633_9595', '2623_9621', '2640_9594', '2622_9606', '2632_9593', '2640_9598', '2660_9572', '2641_9587', '2628_9615', '2624_9618', '2616_9619', '2616_9588', '2628_9602', '2640_9591', '2611_9602', '2641_9590', '2621_9588', '2622_9590', '2630_9603', '2631_9605', '2600_9621', '2615_9617', '2640_9595', '2640_9599', '2615_9610', '2660_9538', '2640_9620', '2629_9579', '2640_9606', '2624_9594', '2628_9586', '2628_9603', '2624_9612', '2626_9595', '2624_9596', '2630_9602', '2628_9583', '2640_9610', '2628_9595', '2617_9606', '2622_9588', '2624_9599', '2622_9603', '2640_9611', '2628_9616', '2624_9595', '2624_9606', '2624_9589', '2611_9619', '2616_9606', '2628_9610', '2628_9582', '2628_9584', '2640_9588']
            self.val_files = ['2640_9586', '2622_9589', '2617_9610', '2628_9577', '2624_9592', '2624_9616', '2629_9608', '2624_9615', '2614_9627', '2621_9590', '2624_9591', '2641_9584', '2621_9602', '2624_9593', '2640_9587', '2628_9604', '2624_9600', '2617_9614', '2628_9581', '2624_9602', '2641_9621', '2640_9584', '2629_9598', '2624_9613']
            self.test_files = ['2624_9611', '2628_9576', '2641_9585', '2628_9579', '2628_9597', '2645_9582', '2628_9592', '2607_9589', '2628_9580', '2614_9600', '2614_9628', '2631_9608', '2635_9596', '2641_9611', '2614_9601', '2624_9588', '2641_9583', '2624_9590', '2631_9609', '2640_9607', '2628_9596', '2625_9598', '2641_9593', '2640_9583']

        if args.add_normals:
            additional_information += '_normals'

        if args.cb_focal_loss:
            self.cb_focal_loss = True
            additional_information += '_cb_focal_loss'
        else: 
            self.cb_focal_loss = False

        self.additional_information = additional_information
        self.all_files = in_files

        # Initiate containers
        self.test_proj = []
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_features = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(in_folder)

    def load_sub_sampled_clouds(self, in_folder):
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            full_name = file_path.split('/')[-1][:-4]
            cloud_name = full_name[:9]

            if self.mode == 'train':
                if cloud_name in self.train_files:
                    cloud_split = 'training'
                elif full_name in self.val_files:
                    cloud_split = 'validation'
                else:
                    continue
            
            elif self.mode == 'test':
                if full_name in self.test_files:
                    cloud_split = 'test'
                else:
                    continue

            base_folder = 'dataset_input/'
            addition = f'input_{cfg.sub_grid_size:.3f}/'
            if len(full_name) > 9:
                if 'ROTATION' in full_name:
                    in_folder = base_folder + 'train_tiles_rotation/0/' + addition
                elif 'TRANSLATION' in full_name:
                    in_folder = base_folder + 'train_tiles_translation/0/' + addition
                elif 'SCALING' in full_name:
                    in_folder = base_folder + 'train_tiles_scaling/0/' + addition
                elif 'CROP' in full_name:
                    in_folder = base_folder + 'train_tiles_crop/0/' + addition
            else:
                in_folder = base_folder + 'train_tiles_regular/0/' + addition

            # Name of the input files
            kd_tree_file = join(
                in_folder, '{:s}_KDTree.pkl'.format(full_name))
            sub_ply_file = join(in_folder, '{:s}.npz'.format(full_name))

            data = np.load(sub_ply_file)
            if args.add_normals:
                sub_features = np.vstack((data['red'], data['green'],
                                        data['blue'], data['intensity'],
                                        data['normal_x'], data['normal_y'], data['normal_z'])).T
            else:
                sub_features = np.vstack((data['red'], data['green'],
                                        data['blue'], data['intensity'])).T
            if cloud_split in ['training', 'validation']:
                sub_labels = data['label'].squeeze()

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_features[cloud_split] += [sub_features]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

            size = sub_features.shape[0] * 4 * 7 * 1e-6
            fname = kd_tree_file.split('/')[-1]
            print(f'{fname} {size:.1f} MB loaded in {time.time() - t0:.1f}s')

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            full_name = file_path.split('/')[-1][:-4]
            cloud_name = full_name[:9]

            if len(full_name) > 9:
                if 'ROTATION' in full_name:
                    in_folder = base_folder + 'train_tiles_rotation/0/' + addition
                elif 'TRANSLATION' in full_name:
                    in_folder = base_folder + 'train_tiles_translation/0/' + addition
                elif 'SCALING' in full_name:
                    in_folder = base_folder + 'train_tiles_scaling/0/' + addition
                elif 'CROP' in full_name:
                    in_folder = base_folder + 'train_tiles_crop/0/' + addition
            else:
                in_folder = base_folder + 'train_tiles_regular/0/' + addition

            # Test projection
            if full_name in self.test_files and self.mode == 'test':
                proj_file = join(in_folder, '{:s}_proj.pkl'.format(full_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]

            # Validation projection and labels
            elif full_name in self.val_files:
                proj_file = join(in_folder, '{:s}_proj.pkl'.format(full_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]  # TODO used?
                print(f'{cloud_name} done in {time.time() - t0:.1f}s')

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        elif split == 'test':
            num_per_epoch = cfg.test_steps * cfg.test_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []

        # Random initialize
        for i, tree in enumerate(self.input_features[split]):
            self.possibility[split] +=\
                                [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] +=\
                                [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud
                # as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(
                        self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10,
                                         size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less
                # than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = (self.input_trees[split][cloud_idx]
                                   .query(pick_point, k=len(points))[1][0])
                else:
                    # Query the predefined number of points
                    queried_idx = (self.input_trees[split][cloud_idx]
                                   .query(pick_point, k=cfg.num_points)[1][0])

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors =\
                    self.input_features[split][cloud_idx][queried_idx]
                if split == 'test':
                    queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                else:
                    queried_pc_labels =\
                        self.input_labels[split][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square(
                                    (points[queried_idx] - pick_point)
                                    .astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] =\
                    float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    (queried_pc_xyz, queried_pc_colors,
                     queried_idx, queried_pc_labels) =\
                        DP.data_aug(queried_pc_xyz, queried_pc_colors,
                                    queried_pc_labels, queried_idx,
                                    cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        if args.add_normals:
            gen_shapes = ([None, 3], [None, 7], [None], [None], [None])
        else:
            gen_shapes = ([None, 3], [None, 4], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels,
                   batch_pc_idx, batch_cloud_idx):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            neighbor_last = tf.numpy_function(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
            for i in range(cfg.num_layers):
                neighbour_idx = tf.numpy_function(DP.knn_search, # ADJUSTED THIS LINE OF CODE
                                           [batch_xyz, batch_xyz, cfg.k_n],
                                           tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1]
                                       // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1]
                                       // cfg.sub_sampling_ratio[i], :]
                up_i = tf.numpy_function( # ADJUSTED THIS LINE OF CODE
                        DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = (input_points + input_neighbors
                          + input_pools + input_up_samples)
            input_list += [batch_features, batch_labels,
                           batch_pc_idx, batch_cloud_idx, neighbor_last]

            return input_list

        return tf_map

    def init_input_pipeline_train(self):
        print('Initiating input pipelines for train and validation')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label]
                                  for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(
                                    gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(
                                    gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(
                                        self.batch_train_data.output_types,
                                        self.batch_train_data.output_shapes)

        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)

    def init_input_pipeline_test(self):
        print('Initiating input pipelines for testing')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label]
                                  for ign_label in self.ignored_labels]
        gen_function_test, gen_types, gen_shapes = self.get_batch_gen('test')
        self.test_data = tf.data.Dataset.from_generator(
                                    gen_function_test, gen_types, gen_shapes)

        self.batch_test_data = self.test_data.batch(cfg.test_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_test_data = self.batch_test_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.test_batch_size)

        iter = tf.data.Iterator.from_structure(
                                        self.batch_test_data.output_types,
                                        self.batch_test_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.test_init_op = iter.make_initializer(self.batch_test_data)


def train(args, in_folder, in_files):
    dataset = Amsterdam3D(args.mode, in_folder, in_files)
    dataset.init_input_pipeline_train()

    if args.model == "RandLANet":
        model = RandLANet(dataset, cfg, args.resume, args.resume_path)
    if args.model == "SCFNet":
        model = SCFNet(dataset, cfg, args.resume, args.resume_path)
    if args.model == "CGANet":
        model = CGANet(dataset, cfg, args.resume, args.resume_path)
    model.train(dataset)


def test(args, in_folder, in_files):
    dataset = Amsterdam3D(args.mode, in_folder, in_files)
    dataset.init_input_pipeline_test()

    cfg.saving = False

    if args.model == "RandLANet":
        model = RandLANet(dataset, cfg)
    if args.model == "SCFNet":
        model = SCFNet(dataset, cfg)
    if args.model == "CGANet":
        model = CGANet(dataset, cfg)

    if args.snap_folder is not None:
        snap_steps = [int(f[:-5].split('-')[-1])
                      for f in os.listdir(args.snap_folder) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(args.snap_folder, 'snap-{:d}'.format(chosen_step))

    else:
        logs = np.sort([os.path.join('results/' + str(args.model), f)
                        for f in os.listdir('results/' + str(args.model)) if f.startswith('Log')])
        chosen_folder = logs[-1]
        snap_path = join(chosen_folder, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1])
                      for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))

    # Code to write predicted files
    tester = ModelTester(model, dataset, args.model, restore_snap=chosen_snap)
    tester.test(model, dataset, args.out_folder)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RandLA-Net and SCFNet implementation.')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--in_folder', metavar='path', action='store',
                        type=str, required=True)
    parser.add_argument('--out_folder', metavar='path', action='store',
                        type=str, required=False)
    parser.add_argument('--snap_folder', metavar='path', action='store',
                        type=str, required=False)
    parser.add_argument('--resume', action='store_true', required=False)
    parser.add_argument('--resume_path', metavar='path', action='store',
                        type=str, required=False)
    parser.add_argument('--model', type=str, required=True, default='RandLANet')
    parser.add_argument('--add_normals', action='store_true', required=False)
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--add_augmented_tiles', action='store_true', required=False)
    parser.add_argument('--augmentation_type', type=str, required=False)
    parser.add_argument('--cb_focal_loss', action='store_true', required=False)
    args = parser.parse_args()
    
    if str(args.model) == 'RandLANet':
        cfg = ConfigAmsterdam3D_RandLANet
    elif str(args.model) == 'SCFNet':
        cfg = ConfigAmsterdam3D_SCFNet
    elif str(args.model) == 'CGANet':
        cfg = ConfigAmsterdam3D_CGANet
    elif str(args.model) == 'GANet':
        cfg = ConfigAmsterdam3D_GANet
    else:
        print("Please provide correct model name. Either RandLANet, SCFNet, CGANet or GANet. Aborting...")
        sys.exit(1)

    # in_folder = join(args.in_folder, f'input_{cfg.sub_grid_size:.3f}')
    in_folder = args.in_folder
    if not os.path.isdir(args.in_folder):
        print(f'The input folder {in_folder} does not exist. Aborting...')
        sys.exit(1)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    addition = f'input_{cfg.sub_grid_size:.3f}'
    regular_files = glob.glob(join(in_folder + '/train_tiles_regular/0/' + addition, '*.npz'))
    if args.add_augmented_tiles:
        if args.augmentation_type == 'ROTATION':
            in_files = regular_files + glob.glob(join(in_folder + '/train_tiles_rotation/0/' + addition, '*.npz'))
        elif args.augmentation_type == 'TRANSLATION':
            in_files = regular_files + glob.glob(join(in_folder + '/train_tiles_translation/0/' + addition, '*.npz'))
        elif args.augmentation_type == 'SCALING':
            in_files = regular_files + glob.glob(join(in_folder + '/train_tiles_scaling/0/' + addition, '*.npz'))
        elif args.augmentation_type == 'CROP':
            in_files = regular_files + glob.glob(join(in_folder + '/train_tiles_crop/0/' + addition, '*.npz'))

    else:
        in_files = regular_files

    if args.mode == 'train':
        train(args, in_folder, in_files)
    elif args.mode == 'test':
        if args.resume:
            done_tiles = utils.get_tilecodes_from_folder(
                                    args.out_folder, extension='.laz')
            test_files = [f for f in in_files
                          if utils.get_tilecode_from_filename(f) not in
                          done_tiles]
        else:
            test_files = in_files

        if args.out_folder:
            if not os.path.isdir(args.out_folder):
                os.makedirs(args.out_folder)

            print(f'Starting inference for {len(test_files)} files...')
            test(args, in_folder, test_files)
        else:
            print('Please provide the output folder. Aborting...')
            sys.exit(1)
    else:
        print('Mode not implemented. Aborting...')
        sys.exit(1)
