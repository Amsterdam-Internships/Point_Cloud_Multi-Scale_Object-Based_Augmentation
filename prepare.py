from sklearn.neighbors import KDTree
from os.path import join
import numpy as np
import os
import sys
import glob
import pickle
import argparse
import laspy
from tqdm import tqdm

from core.helpers.helper_ply import write_ply
from core.helpers.helper_tool import DataProcessing as DP
from configs.config_Amsterdam3D import ConfigAmsterdam3D as cfg
import core.helpers.helper_filelists as utils

def run(args):
    in_files = glob.glob(join(args.in_folder, '*.laz'))

    if args.resume:
        if args.save_ply:
            save_extension = '.ply'
        else:
            save_extension = '.npz'

        # Check if there are folders
        sub_folders = [name for name in os.listdir(args.out_folder)
                       if os.path.isdir(os.path.join(args.out_folder, name))]

        # Find the last folder and based on that define the new sub folder
        last_folder = sorted(list(map(int, sub_folders)))[-1]
        sub_folder = last_folder + 1

        print(f'Resume with new folder {sub_folder}.')

        done_tiles = utils.get_tilecodes_from_folder_nested(
                                args.out_folder, extension=save_extension)
        files = [f for f in in_files
                 if utils.get_tilecode_from_filename(f) not in done_tiles]
    else:
        sub_folder = 0
        files = in_files

    # Create the sub folder
    out_folder = join(args.out_folder, str(sub_folder),
                      f'input_{cfg.sub_grid_size:.3f}')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Initialize
    size_of_files = 0

    files_tqdm = tqdm(files, unit='file', smoothing=0)
    for f in files_tqdm:
        base_filename = os.path.splitext(os.path.basename(f))[0]
        print(base_filename)

        data = laspy.read(f)
        # Load points
        xyz = (np.vstack((data.x - cfg.x_offset, data.y - cfg.y_offset,
                         data.z)).T.astype(np.float32))
        # Load RGB
        rgb = np.vstack((data.red, data.green, data.blue)).T
        # Check color depth since this somehow seems to differ
        max_c = np.max(rgb)
        if max_c <= 1.:
            pass
        elif max_c < 2**8:
            rgb = rgb / 255.
        elif max_c < 2**16:
            rgb = rgb / 65535.
        else:
            print('RGB more than 16 bits, not implemented. Aborting...')
            sys.exit(1)
        rgb = rgb.astype(np.float16)
        # Load intensity
        intensity = ((data.intensity / 65535.)
                     .reshape((-1, 1)).astype(np.float16))

        if args.add_normals:
            normals = np.vstack((data.normal_x, data.normal_y, data.normal_z)).T
            normals = normals.astype(np.float16)
            features = np.hstack((rgb, intensity, normals))
        else:
            features = np.hstack((rgb, intensity))

        if args.mode == 'train':
            labels = data.label
            # Ignore label 0 (unlabeled)
            mask = ((labels == 0))
            labels[mask] = 99

            # Move label 10 (road) to index 0, so that we start with label index 0
            labels[labels == 10] = 0

            mask = (labels != 99)
            labels_prepared = labels[mask].astype(np.uint8).reshape((-1,))

            sub_xyz, sub_features, sub_labels =\
                DP.grid_sub_sampling(xyz[mask], features[mask],
                                     labels_prepared,
                                     grid_size=cfg.sub_grid_size)

            if args.save_ply:
                # NOTE: Save the <x,y,z> to view the result in CloudCompare.
                sub_ply_file = join(out_folder, base_filename + '.ply')
                if args.add_normals:
                    write_ply(sub_ply_file, [sub_xyz, sub_features, sub_labels],
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity', 'normal_x', 'normal_y', 'normal_z',
                            'label'])
                else: 
                    write_ply(sub_ply_file, [sub_xyz, sub_features, sub_labels],
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity',
                            'label'])
            else:
                sub_npz_file = join(out_folder, base_filename + '.npz')
                if args.add_normals:
                    np.savez_compressed(sub_npz_file, red=sub_features[:, 0],
                                        green=sub_features[:, 1],
                                        blue=sub_features[:, 2],
                                        intensity=sub_features[:, 3],
                                        normal_x=sub_features[:, 4],
                                        normal_y=sub_features[:, 5],
                                        normal_z=sub_features[:, 6],
                                        label=sub_labels)
                else:
                    np.savez_compressed(sub_npz_file, red=sub_features[:, 0],
                                        green=sub_features[:, 1],
                                        blue=sub_features[:, 2],
                                        intensity=sub_features[:, 3],
                                        label=sub_labels)

        elif args.mode == 'test':
            labels = np.zeros((len(data.x),), dtype=np.uint8)
            sub_xyz, sub_features = DP.grid_sub_sampling(
                                        xyz, features,
                                        grid_size=cfg.sub_grid_size)

            if args.save_ply:
                sub_ply_file = join(out_folder, base_filename + '.ply')
                if args.add_normals:
                    write_ply(sub_ply_file, [sub_xyz, sub_features],
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity', 'normal_x', 'normal_y', 'normal_z'])
                else:
                     write_ply(sub_ply_file, [sub_xyz, sub_features],
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity'])
            else:
                sub_npz_file = join(out_folder, base_filename + '.npz')
                if args.add_normals:
                    np.savez_compressed(sub_npz_file, red=sub_features[:, 0],
                                        green=sub_features[:, 1],
                                        blue=sub_features[:, 2],
                                        intensity=sub_features[:, 3],
                                        normal_x=sub_features[:, 4],
                                        normal_y=sub_features[:, 5],
                                        normal_z=sub_features[:, 6])
                else:
                    np.savez_compressed(sub_npz_file, red=sub_features[:, 0],
                                        green=sub_features[:, 1],
                                        blue=sub_features[:, 2],
                                        intensity=sub_features[:, 3])

        # save sub_cloud and KDTree file
        search_tree = KDTree(sub_xyz)
        kd_tree_file = join(out_folder, base_filename + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(out_folder, base_filename + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)

        # Calculate the size of all processed files in Bytes
        if args.save_ply:
            size_of_files += os.path.getsize(sub_ply_file)
        else:
            size_of_files += os.path.getsize(sub_npz_file)

        size_of_files += os.path.getsize(kd_tree_file)
        size_of_files += os.path.getsize(proj_save)

        if size_of_files > cfg.max_size_bytes:
            # Different folder
            sub_folder += 1
            # Reset the size_of_files
            size_of_files = 0

            out_folder = join(args.out_folder, str(sub_folder),
                            f'input_{cfg.sub_grid_size:.3f}')
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for RandLA-Net.')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--in_folder', metavar='path', type=str, required=True)
    parser.add_argument('--out_folder', metavar='path', type=str,
                        required=True)
    parser.add_argument('--resume', action='store_true')
    # save_ply -> Useful for debugging sub Point Cloud in CloudCompare.
    parser.add_argument('--save_ply', action='store_true')
    parser.add_argument('--add_normals', action='store_true')

    args = parser.parse_args()

    if not os.path.isdir(args.in_folder):
        print('The input path does not exist')
        sys.exit(1)

    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)

    run(args)