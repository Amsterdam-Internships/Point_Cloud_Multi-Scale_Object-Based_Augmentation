import pathlib
import re
import os
import laspy
from tqdm import tqdm


def get_tilecode_from_filename(filename):
    """Extract the tile code from a file name."""
    return re.match(r'.*(\d{4}_\d{4}).*', filename)[1]


def get_tilecodes_from_folder_nested(folder, extension=''):
    """Get a set of unique tilecodes for the LAS files in a given folder."""
    files = pathlib.Path(folder).rglob(f'*{extension}')
    tilecodes = set([get_tilecode_from_filename(file.name) for file in files])
    return tilecodes


def get_tilecodes_from_folder(folder, prefix='', extension=''):
    """Get a set of unique tilecodes for the LAS files in a given folder."""
    files = pathlib.Path(folder).glob(f'{prefix}*{extension}')
    tilecodes = set([get_tilecode_from_filename(file.name) for file in files])
    print(tilecodes)
    print('')
    return tilecodes


def merge_cloud_pred(cloud_file, pred_file, out_file, label_dict=None):
    """Merge predicted labels into a point cloud LAS file."""
    cloud = laspy.read(cloud_file)
    pred = laspy.read(pred_file)

    if len(pred.label) != len(cloud.x):
        print('Dimension mismatch between cloud and prediction '
              + f'for tile {get_tilecode_from_filename(cloud)}.')
        return

    if 'label' not in cloud.point_format.extra_dimension_names:
        cloud.add_extra_dim(laspy.ExtraBytesParams(
                            name="label", type="uint8", description="Labels"))

    cloud.label = pred.label.astype('uint8')
    if label_dict is not None:
        for key, value in label_dict.items():
            cloud.label[cloud.label == key] = value

    if 'probability' in pred.point_format.extra_dimension_names:
        if 'probability' not in cloud.point_format.extra_dimension_names:
            cloud.add_extra_dim(laspy.ExtraBytesParams(
                                name="probability", type="float32",
                                description="Probabilities"))
        cloud.probability = pred.probability

    cloud.write(out_file)


def merge_cloud_pred_folder(cloud_folder, pred_folder, out_folder='',
                            cloud_prefix='refined', pred_prefix='pred',
                            out_prefix='merged', label_dict=None,
                            resume=False, hide_progress=False):
    """
    Merge the labels of all predicted tiles in a folder into the corresponding
    point clouds and save the result.
    Parameters
    ----------
    cloud_folder : str
        Folder containing the unlabelled .laz files.
    pred_folder : str
        Folder containing corresponding .laz files with predicted labels.
    out_folder : str (default: '')
        Folder in which to save the merged clouds.
    cloud_prefix : str (default: 'filtered')
        Prefix of unlabelled .laz files.
    pred_prefix : str (default: 'pred')
        Prefix of predicted .laz files.
    out_prefix : str (default: 'merged')
        Prefix of output files.
    label_dict : dict (optional)
        Mapping from predicted labels to saved labels.
    resume : bool (default: False)
        Skip merge when output file already exists.
    hide_progress : bool (default: False)
        Whether to hide the progress bar.
    """
    cloud_codes = get_tilecodes_from_folder(cloud_folder, cloud_prefix)
    pred_codes = get_tilecodes_from_folder(pred_folder, pred_prefix)
    in_codes = cloud_codes.intersection(pred_codes)
    if resume:
        done_codes = get_tilecodes_from_folder(out_folder, out_prefix)
        todo_codes = {c for c in in_codes if c not in done_codes}
    else:
        todo_codes = in_codes
    files_tqdm = tqdm(todo_codes, unit="file", disable=hide_progress,
                      smoothing=0)
    print(f'{len(todo_codes)} files found.')

    for tilecode in files_tqdm:
        files_tqdm.set_postfix_str(tilecode)
        print(f'Processing tile {tilecode}...')
        cloud_file = os.path.join(
                        cloud_folder, cloud_prefix + '_' + tilecode + '.laz')
        pred_file = os.path.join(
                        pred_folder, pred_prefix + '_' + tilecode + '.laz')
        out_file = os.path.join(
                        out_folder, out_prefix + '_' + tilecode + '.laz')
        merge_cloud_pred(cloud_file, pred_file, out_file, label_dict)
