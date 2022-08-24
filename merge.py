import argparse
import os
import sys

import core.helpers.helper_filelists as helper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine predictions with '
                                     + 'original clouds.')
    parser.add_argument('--cloud_folder', metavar='path', type=str,
                        required=True)
    parser.add_argument('--pred_folder', metavar='path', type=str,
                        required=True)
    parser.add_argument('--out_folder', metavar='path', type=str,
                        required=True)

    args = parser.parse_args()

    # if not os.path.isdir(args.cloud_folder):
    #     print('The input folder(s) does not exist')
    #     sys.exit(1)

    # if not os.path.isdir(args.pred_folder):
    #     print('fdafd The input folder(s) does not exist')
    #     sys.exit(1)

    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)

    helper.merge_cloud_pred_folder(args.cloud_folder, args.pred_folder,
                                   out_folder=args.out_folder)