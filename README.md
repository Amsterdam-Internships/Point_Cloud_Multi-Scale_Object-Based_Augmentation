# MOBA: Multi-Scale Object-Based Augmentation for Imbalanced Point Clouds

This repository contains Tensorflow implementations of [RandLANet](http://arxiv.org/abs/1911.11236), [SCFNet](https://ieeexplore.ieee.org/document/9577763) and [RandLA-Net with CGA](https://openaccess.thecvf.com/content/CVPR2021/html/Lu_CGA-Net_Category_Guided_Aggregation_for_Point_Cloud_Semantic_Segmentation_CVPR_2021_paper.html) with small improvements to the [original RandLANet](https://github.com/QingyongHu/RandLA-Net), [original SCFNet](https://github.com/leofansq/SCF-Net) and [original RandLANet with CGA](https://github.com/MCG-NJU/CGA-Net) implementations. This repository only supports the 3D Point Cloud licensed to City of Amsterdam.

## Preparation
This code has been tested with Python 3.7, Tensorflow 1.15.5, CUDA 11.2 on Ubuntu 18.04.

1. Clone this repository

  ```sh
  git clone https://github.com/Amsterdam-AI-Team/RandLA-Net.git
  ```

2. Install all Python dependencies

  ```sh
  cd RandLA-Net
  pip install -r requirements.txt
  ```

### Dataset
The City of Amsterdam acquired a street-level 3D Point Cloud encompassing the entire municipal area. The point cloud, provided by Cyclomedia Technology, is recorded using both a panoramic image capturing device together with a Velodyne HDL-32 LiDAR sensor, resulting in a point cloud that includes not only <x, y, z> coordinates, but also RGB and intensity data. The 3D Point Cloud is licensed, but it may become open in the future. An example tile is available [here](https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing/tree/main/datasets/pointcloud).


## Usage
- Prepare the dataset to train, evaluate and test a model
  ```sh
  python3 prepare.py --mode 'train' --in_folder 'raw_files/' --out_folder 'dataset_input/train_npz/'
   ```
   ```sh
  python3 prepare.py --mode 'test' --in_folder 'raw_files/' --out_folder 'dataset_input/test_npz/'
   ```
  The following command line arguments are available for preparation:
   - `--in_folder` is the folder with the raw point cloud files.
   - `--out_folder` is the folder to write the prepared point cloud files to.

- Train a model

  ```sh
  python3 main.py --mode 'train' --model "RandLANet" --in_folder 'dataset_input/train_npz/0/' 
   ```
  The following command line arguments are available training:
  - `--model` can be 'RandLANet','SCFNet' or 'RandLANet_CGA', currently.
  - `--in_folder` is the folder with prepared train files.
  - `--resume` is an optional argument that allows to continue training a model that has already been trained for 1 or more epochs.
  - `--resume_path` is an optional argument that specifies the folder containing the model to continue training if `--resume` is passed as argument. If `--resume_path` is not passed, it will continue training on the last trained model.
  
- Evaluate a trained model

  ```sh
  python3 main.py --mode 'evaluate_only' --model "RandLANet" --in_folder 'dataset_input/train_npz/0/' --resume --resume_path 'path/to/snapshots/trained/model/'
   ```
  The following command line arguments are available training:
  - `--model` can be 'RandLANet','SCFNet' or 'RandLANet_CGA', currently.
  - `--in_folder` is the folder with prepared train files.
  - `--resume` is a required argument that ensures we can select a trained model.
  - `--resume_path` is an optional argument that specifies the folder containing the model to evaluate. If `--resume_path` is not passed, it will evaluate the last trained model.

- Test a model

  ```sh
  python3 main.py --mode 'test' --model "RandLANet" --in_folder 'dataset_input/test_npz/0/' --out_folder 'dataset_input/predicted_laz/RandLANet/0/'
   ```
  The following command line arguments are available testing:
  - `--model` can be 'RandLANet', 'SCFNet' or 'RandLANet', currently.
  - `--in_folder` is the folder with prepared test files.
  - `--out_folder` specifies the folder to write the files with predicted segmentations to.
  - `--snap_folder` is an optional argument that specifies the folder containing the model to test. If `--snap_folder` is not passed, it will test the trained model that was created the latest.
  - `--resume` is an optional argument that allows to skip test files in `--out_folder` that have already been predicted.

- Merge predictions from tested model with original Point Cloud tiles.

  ```sh
  python3 merge.py --cloud_folder 'raw_files/' --pred_folder 'dataset_input/predicted_laz/RandLANet/0/' --out_folder 'merged_point_clouds/RandLANet/0/'
  ```
  The following command line arguments should be passed for merging:
  - `--cloud_folder` is the folder with the raw point cloud files.
  - `--pred_folder` is the folder with the predicted segmentation of point cloud files.
  - `--out_folder` is the folder to write the merged point clouds to.

### Visualization

One can visualize the evolution of the loss with Tensorboard.

On a separate terminal, launch:

  ```sh
  tensorboard --logdir tensorflow_logs
  ```

## Citation

This work implements the work presented in [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](http://arxiv.org/abs/1911.11236), [SCF-Net: Learning Spatial Contextual Features for Large-Scale Point Cloud Segmentation](https://ieeexplore.ieee.org/document/9577763) and [CGA-Net: Category Guided Aggregation for Point Cloud Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/Lu_CGA-Net_Category_Guided_Aggregation_for_Point_Cloud_Semantic_Segmentation_CVPR_2021_paper.html).

To cite the original papers:
  ```
  @article{RandLA-Net,
    arxivId = {1911.11236},
    author = {Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
    eprint = {1911.11236},
    title = {{RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds}},
    url = {http://arxiv.org/abs/1911.11236},
    year = {2019}
  }
  ```
  ```
  @inproceedings{fan2021scf,
    title={SCF-Net: Learning Spatial Contextual Features for Large-Scale Point Cloud Segmentation},
    author={Fan, Siqi and Dong, Qiulei and Zhu, Fenghua and Lv, Yisheng and Ye, Peijun and Wang, Fei-Yue},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={14504--14513},
    year={2021}
  }
  ```
  ```
  @inproceedings{lu2021cga,
    title={CGA-Net: Category Guided Aggregation for Point Cloud Semantic Segmentation},
    author={Lu, Tao and Wang, Limin and Wu, Gangshan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={11693--11702},
    year={2021}
  }
  ```
