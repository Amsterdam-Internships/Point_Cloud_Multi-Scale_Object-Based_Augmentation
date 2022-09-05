# MOBA: Multi-Scale Object-Based Augmentation for Imbalanced Point Clouds

This repository contains the code utilized and writting as part of the MSc Artificial Intelligence Thesis at the UvA in collaboration with the City of Amsterdam. The aim of this project was to improve the mapping of underrepresented urban topographical object properties in imbalanced urban point clouds using deep-learning semantic segmentation models. From such urban mappings, useful insights can be gathered. For example, information on the region-based density of trash cans in neighborhoods or misalignment of traffic lights on roadsides can be collected. To this end, Multi-Scale Object-Based Augmentation (MOBA) was introduced. MOBA is a memory-efficient augmentation technique that crops point clouds around minority objects using multiple radii, enabling models to learn better minority class features (i.e., features of underrepresented urban topographical object properties) from both local and global point cloud information. An example of MOBA is given below.


<p align="center"> <img src="https://i.ibb.co/rMLmcV6/moba.png" width="800%"> </p>


Besides the code for MOBA, this repository contains modified Tensorflow implementations of [RandLANet](http://arxiv.org/abs/1911.11236), [SCFNet](https://ieeexplore.ieee.org/document/9577763) and [RandLA-Net with CGA](https://openaccess.thecvf.com/content/CVPR2021/html/Lu_CGA-Net_Category_Guided_Aggregation_for_Point_Cloud_Semantic_Segmentation_CVPR_2021_paper.html) to test MOBA and compare MOBA to benchmark augmentation methods. Furhtermore, this repository is designed to support the 3D Point Cloud licensed to City of Amsterdam, but can be used for other point cloud datasets as well.

## Preparation
This code has been tested with Python 3.7, Tensorflow 1.15.5, CUDA 11.2 on Ubuntu 18.04.

1. Clone this repository

  ```sh
  git clone https://github.com/Amsterdam-Internships/Point_Cloud_Multi-Scale_Object-Based_Augmentation.git
  ```

2. Install all Python dependencies

  ```sh
  pip install -r requirements.txt
  ```

### Dataset
The City of Amsterdam acquired a street-level 3D Point Cloud encompassing the entire municipal area. The point cloud, provided by Cyclomedia Technology, is recorded using both a panoramic image capturing device together with a Velodyne HDL-32 LiDAR sensor, resulting in a point cloud that includes not only <x, y, z> coordinates, but also RGB and intensity data. The 3D Point Cloud is licensed, but it may become open in the future. An example tile is available [here](https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing/tree/main/datasets/pointcloud).


## Usage
- Generate augmented point clouds with MOBA (or a benchmark augmentation method)
  ```sh
  python3 generate_augmented_pc.py
   ```

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
   - `--add_normals` is an optional argument that includes normals as features to the prepared files. These normals need to be in the raw files and can be added with the ```add_normals_to_pc.py``` file.

- Train a model

  ```sh
  python3 main.py --mode 'train' --model "RandLANet" --in_folder 'dataset_input/train_npz/0/' 
   ```
  The following command line arguments are available training:
  - `--model` can be 'RandLANet','SCFNet' or 'CGANet', currently.
  - `--in_folder` is the folder with prepared train files.
  - `--resume` is an optional argument that allows to continue training a model that has already been trained for 1 or more epochs.
  - `--resume_path` is an optional argument that specifies the folder containing the model to continue training if `--resume` is passed as argument. If `--resume_path` is not passed, it will continue training on the last trained model.
  - `--add_normals` is an optional argument that allows to train models with normals as features. Tese normals should be included in the prepared point clouds.
  - `--split` sets the train/val/test split that is used. Currently, this can be set as either 1,2 or 3.
  - `--add_augmented_tiles` is an optional argument that needs to be set to train models on additional augmented tiles.
  - `--augmentation_type` sets the type of augmentation the model can be trained on. Currently, this is either MOBA, ROTATION, TRANSLATION or SCALING.
  - `--cb_focal_loss` is an optional argument that allows to train the model with Class-Balanced Focal Loss.
  
- Test a model

  ```sh
  python3 main.py --mode 'test' --model "RandLANet" --in_folder 'dataset_input/test_npz/0/' --out_folder 'dataset_input/predicted_laz/RandLANet/0/'
   ```
  The following command line arguments are available testing:
  - `--model` can be 'RandLANet', 'SCFNet' or 'CGANet', currently.
  - `--in_folder` is the folder with prepared test files.
  - `--out_folder` specifies the folder to write the files with predicted segmentations to.
  - `--snap_folder` is an optional argument that specifies the folder containing the model to test. If `--snap_folder` is not passed, it will test the trained model that was created the latest.
  - `--resume` is an optional argument that allows to skip test files in `--out_folder` that have already been predicted.
  - `--add_normals` is an optional argument that allows to train models with normals as features. Tese normals should be included in the prepared point clouds.
  - `--split` sets the train/val/test split that is used. Currently, this can be set as either 1,2 or 3.

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
