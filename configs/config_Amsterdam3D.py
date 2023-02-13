"""RandLA-Net configuration for Amsterdam point clouds."""

class ConfigAmsterdam3D:
    # Dataset configuration
    labels = {0: 'Unlabelled',
              1: 'Ground',
              2: 'Building',
              3: 'Tree',
              4: 'Street light',
              5: 'Traffic sign',
              6: 'Traffic light',
              7: 'Car',
              8: 'City bench',
              9: 'Rubbish bin',
              10: 'Road',
              99: 'Noise'}

    # idx_to_label = {0: 7}  # Set car back to label 7 when merging predictions (with 6 classes)
    idx_to_label = {0: 10}  # Set road back to label 10 when merging predictions (with 10 classes)
    inference_on_labels = []

    # Use an offset to subtract from the raw coordinates. Otherwise, use 0.
    x_offset = 129500
    y_offset = 476500
                        
    max_size_bytes = 12000000000 # Approximately max size Bytes to load for 16GB GPU
    sub_grid_size = 0.04  # preprocess_parameter


class ConfigAmsterdam3D_RandLANet:
    # Dataset configuration
    model = 'RandLANet'
    num_classes = 10  # Number of valid classes

    # Number of points per class.
    class_weights = [142165464, 116393327, 121356229, 26521378, 1050837,
                    464230, 218948, 16990894, 156477, 65832]

    # RandLA-Net configuration.
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 3  # batch_size during training
    train_steps = 1000  # Number of steps per epochs
    val_batch_size = 10  # batch_size during validation
    val_steps = 400 # Number of validation steps per epoch
    test_batch_size = 10  # batch_size during inference
    test_steps = 400 # batch_size during testing

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # Sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter

    # Training configuration.
    max_epoch = 125  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'tensorflow_logs/RandLANet/10_classes'
    saving = True
    saving_path = None


class ConfigAmsterdam3D_SCFNet:
    # Dataset configuration.
    model = 'SCFNet'
    num_classes = 10  # Number of valid classes

    # Number of points per class.
    class_weights = [142165464, 116393327, 121356229, 26521378, 1050837, 
                    464230, 218948, 16990894, 156477, 65832]

    # SCFNet configuration.
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 3  # batch_size during training
    train_steps = 1000  # Number of steps per epochs
    val_batch_size = 10  # batch_size during validation
    val_steps = 400 # Number of validation steps per epoch
    test_batch_size = 10  # batch_size during inference
    test_steps = 400 # batch_size during testing

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # Sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter

    # Training configuration.
    max_epoch = 125  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'tensorflow_logs/SCFNet/10_classes'
    saving = True
    saving_path = None


class ConfigAmsterdam3D_CGANet:
    # Dataset configuration.
    model = 'CGANet'
    num_classes = 10  # Number of valid classes

    # Number of points per class.
    class_weights = [142165464, 116393327, 121356229, 26521378, 1050837, 
                    464230, 218948, 16990894, 156477, 65832]

    # CGANet configuration.
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 3  # batch_size during training
    train_steps = 1000  # Number of steps per epochs
    val_batch_size = 10  # batch_size during validation
    val_steps = 400 # Number of validation steps per epoch
    test_batch_size = 10  # batch_size during inference
    test_steps = 400 # batch_size during testing

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # Sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter

    # Training configuration.
    max_epoch = 125  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'tensorflow_logs/CGANet/10_classes'
    saving = True
    saving_path = None


class ConfigAmsterdam3D_GANet:
    # Dataset configuration
    model = 'GANet'
    num_classes = 10  # Number of valid classes

    # Number of points per class.
    class_weights = [142165464, 116393327, 121356229, 26521378, 1050837,
                    464230, 218948, 16990894, 156477, 65832]

    # RandLA-Net configuration.
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 3  # batch_size during training
    train_steps = 1000  # Number of steps per epochs
    val_batch_size = 10  # batch_size during validation
    val_steps = 400 # Number of validation steps per epoch
    test_batch_size = 10  # batch_size during inference
    test_steps = 400 # batch_size during testing

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # Sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter

    # Training configuration.
    max_epoch = 125  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'tensorflow_logs/GANet/10_classes'
    saving = True
    saving_path = None