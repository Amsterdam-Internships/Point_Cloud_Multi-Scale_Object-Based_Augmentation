import tensorflow as tf
import numpy as np
import laspy
from os.path import join, dirname, abspath, splitext
import sys
import time
import os

BASE_DIR = dirname(abspath(__file__))
sys.path.append(BASE_DIR)
from configs.config_Amsterdam3D import ConfigAmsterdam3D as cfg


def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)


def write_las(laz_path, labels, probs=None):
    """
    Saving the point labels and probabilities as a .laz file.
    """
    for key, value in cfg.idx_to_label.items():
        if key != value:
            labels[labels == key] = value

    outfile = laspy.create(file_version="1.2", point_format=3)
    outfile.add_extra_dim(laspy.ExtraBytesParams(
            name="label", type="uint8", description="Labels"))
    outfile.label = np.squeeze(labels)

    if probs is not None:
        outfile.add_extra_dim(laspy.ExtraBytesParams(
            name="probability", type="float32", description="Probabilities"))
        outfile.probability = np.squeeze(probs)
    outfile.write(laz_path)


class ModelTester:
    def __init__(self, model, dataset, model_name, restore_snap=None,):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        log_time = time.strftime('_%Y-%m-%d_%H-%M-%S', time.gmtime())
        if not os.path.isdir('test_logs/' + model_name + '/'):
            os.makedirs('test_logs/' + model_name + '/')
        self.Log_file = open('test_logs/' + model_name + '/log_test_' + model_name + '_' + dataset.name + log_time + '.txt', 'w+')

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        # Load trained model
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)

        # Initiate global prediction over all test clouds
        self.test_probs = [np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
                           for l in dataset.input_trees['test']]

    def test(self, model, dataset, test_path, num_votes=100):
        # Smoothing parameter for votes
        test_smooth = 0.95  # TODO try other smoothing values

        # Initialise iterator with test/test data
        self.sess.run(dataset.test_init_op)

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:
            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],
                       )

                stacked_probs, stacked_labels, point_idx, cloud_idx =\
                    self.sess.run(ops, {model.is_training: False})
                stacked_probs = np.reshape(stacked_probs,
                                           [model.config.test_batch_size,
                                            model.config.num_points,
                                            model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][p_idx] = (
                        test_smooth * self.test_probs[c_i][p_idx]
                        + (1 - test_smooth) * probs)
                step_id += 1

            except tf.errors.OutOfRangeError:

                new_min = np.min(dataset.min_possibility['test'])
                log_out(f'Epoch {epoch_id:3d}, end. ' +
                        f'Min possibility = {new_min:.1f}', self.Log_file)

                if last_min + 1 < new_min:  # TODO maybe use: last_min + 4 < new_min:

                    # Update last_min
                    last_min += 1

                    # files = dataset.test_files
                    files = [file_ for file_ in dataset.all_files if file_.split('/')[-1][:-4] in dataset.test_files]
                    num_val = len(files)

                    if int(np.ceil(new_min)) % 1 == 0:

                        # Project predictions
                        log_out(f'\nReproject Vote #{int(np.floor(new_min))}',
                                self.Log_file)
                        proj_probs_list = []

                        print(f'Reprojecting {num_val} test files.')
                        for i_val, file_path in enumerate(files):
                            print('.', end='')
                            # Reproject probs back to the evaluations points
                            proj_idx = dataset.test_proj[i_val]
                            probs = self.test_probs[i_val][proj_idx, :]
                            proj_probs_list += [probs]
                        print('\nDone!')

                        print(f'Writing predictions for {num_val} test files.')
                        for i_test, file_path in enumerate(files):  # TODO merge with other for loop
                            print('.', end='')
                            # Get the predicted labels
                            preds = (dataset.label_values[
                                np.argmax(proj_probs_list[i_test], axis=1)]
                                     .astype(np.uint8))
                            probs = (np.max(proj_probs_list[i_test], axis=1)
                                     .astype(np.float16))

                            # Save preds
                            cloud_name = splitext(file_path.split('/')[-1])[0].split('filtered')[-1]
                            write_las(join(test_path, 'pred_'+cloud_name+'.laz'),
                                      preds, probs=probs)
                            with open(join(test_path, 'pred_'+cloud_name+'.txt'), 'w') as f:
                                labels = np.squeeze(preds)
                                for label in labels:
                                    if label == 0:
                                        label = 9
                                    f.write(str(label))
                                    f.write('\n')


                        print('\nDone!')
                        self.sess.close()
                        return

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue

        return
