from os.path import exists, join, dirname, abspath
import os
from os import makedirs, listdir
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import sys

BASE_DIR = dirname(abspath(__file__))
sys.path.append(BASE_DIR)

from core.helpers.helper_tool import DataProcessing as DP
import core.helpers.helper_tf_util as helper_tf_util


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config, resume=False, resume_path=None):
        flat_inputs = dataset.flat_inputs
        self.config = config
        additional_information = dataset.additional_information
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                if not os.path.isdir('results/CGANet/{}/'.format(additional_information)):
                    os.makedirs('results/CGANet/{}/'.format(additional_information))
                if resume:
                    self.saving_path = join('results/CGANet/{}/'.format(additional_information), sorted(listdir('results/CGANet/'))[-1])
                else: 
                    self.saving_path = time.strftime('results/CGANet/{}/Log_%Y-%m-%d_%H-%M-%S'.format(additional_information), time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]
            self.inputs['neighbor_last'] = flat_inputs[4 * num_layers + 4]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name, config.class_weights)

    
            log_time = time.strftime('_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not os.path.isdir('train_logs/CGANet/'):
                os.makedirs('train_logs/CGANet/')
            self.Log_file = open('train_logs/CGANet/log_train_CGANet_' + additional_information + log_time + '.txt', 'w+')

        # print configuration 
        log_out('config: {}'.format(self.config.__dict__), self.Log_file)

        with tf.variable_scope('layers'):
            self.logits, self.logits_cga, self.binary = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.logits_cga = tf.reshape(self.logits_cga, [-1, config.num_classes])
            self.binary = tf.reshape(self.binary, [-1, 2])
            temp_label = self.labels
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_logits_cga = tf.gather(self.logits_cga, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            # self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)
            if dataset.cb_focal_loss:
                self.loss = self.get_cb_focal_loss(valid_logits, valid_labels, config.class_weights)
            else:
                self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

            self.loss_cga = self.get_loss(valid_logits_cga, valid_labels, self.class_weights)
            self.loss_weight = tf.get_collection("weight_losses")
            # self.loss_weight = tf.add_n(self.loss_weight, name='losses_weight')

            #### begin binary loss ####
            self.labels = tf.expand_dims(temp_label, axis=-1)
            neighbor_idx = self.inputs['neighbor_last'][:, :, ::2]
            n_neighbors = tf.shape(neighbor_idx)[-1]
            labels_center = tf.tile(self.labels, [1, 1, n_neighbors])
            labels_neighbor = tf.squeeze(self.gather_neighbour(self.labels, neighbor_idx), axis=-1)
            self.binary_label = tf.reshape(tf.cast(tf.equal(labels_center, labels_neighbor), tf.int32), [-1])
            self.loss_binary = self.get_loss(self.binary, self.binary_label)
            #### end binary loss ####

            self.labels = tf.reshape(self.labels, [-1])

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate*(0.95**self.training_epoch), trainable=False, name='learning_rate')
            self.total_loss = self.loss+self.loss_cga+self.loss_binary
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits_cga, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits_cga)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.total_loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if resume:
            if resume_path is None:
                # Assume the last standard saving_path
                resume_path = join('results/CGANet/', sorted(listdir('results/CGANet/'))[-1] + '/')
            print(f'Resuming previous session from [{resume_path}].')
            snap_path = join(resume_path, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1])
                          for f in listdir(snap_path) if f[-5:] == '.meta']
            best_step = np.sort(snap_steps)[-1]
            best_snap = join(snap_path, 'snap-{:d}'.format(best_step))
            self.saver.restore(self.sess, best_snap)
            self.training_step = best_step
            self.training_epoch = int(best_step / self.config.train_steps)
            print(f'Model restored from {best_snap}.')
            print(f'Resuming from epoch {self.training_epoch}, '
                  + f'step {self.training_step}.')

    def inference(self, inputs, is_training):

        d_out = self.config.d_out
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        # ###########################original output############################
        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)

        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])

        # ###########################CGA Output############################
        to_be_augmented = f_decoder_list[-1]
        cga_features, binary = self.cga(inputs['xyz'][0], inputs['neighbor_last'][:,:,::2],
                                                                           to_be_augmented, 32, is_training)

        f_layer_cga_fc1 = helper_tf_util.conv2d(cga_features, 64, [1, 1], 'fc1_cga', [1, 1], 'VALID', True, is_training)
        f_layer_cga_fc2 = helper_tf_util.conv2d(f_layer_cga_fc1, 32, [1, 1], 'fc2_cga', [1, 1], 'VALID', True, is_training)

        f_layer_cga_drop = helper_tf_util.dropout(f_layer_cga_fc2, keep_prob=0.5, is_training=is_training, scope='dp1_cga')
        f_layer_cga_fc3 = helper_tf_util.conv2d(f_layer_cga_drop, self.config.num_classes, [1, 1], 'fc_cga', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out_cga = tf.squeeze(f_layer_cga_fc3, [2])

        return f_out, f_out_cga, binary

    def cga(self, xyz, neighbor_idx, F, fdim, is_training):
        """A head for scene segmentation with resnet backbone.
            Args:
                xyz: BxNx3
                neighbor_idx: BxNxn_neighbor
                F: BxNx1xC
                fdim: int32, the feature dim
                is_training: True indicates training phase
            Returns:
                aggregated features [B, N, C], binary logits [B, N, n_neighbors, 2]
            """
        with tf.variable_scope('feature_augmentation_head') as sc:
            batch_size = tf.shape(xyz)[0]
            N = tf.shape(xyz)[1]
            n_neighbors = tf.shape(neighbor_idx)[-1]
            input_dim = F.get_shape()[-1].value
            features = F

            query_points = xyz
            features = tf.squeeze(features, axis=2)
            neighbor_features = self.gather_neighbour(features, neighbor_idx)
            features = tf.expand_dims(features, axis=2)
            features = tf.tile(features, [1, 1, n_neighbors, 1])
            center_points = tf.expand_dims(query_points, 2)
            neighbor_points = self.gather_neighbour(query_points, neighbor_idx)
            diff_xyz = center_points - neighbor_points

            concat_features_xyz = tf.concat([features, neighbor_features, diff_xyz], axis=-1)
            concat_features_xyz = tf.reshape(concat_features_xyz, [batch_size, N*n_neighbors, input_dim*2+3])
            concat_features_xyz = tf.expand_dims(concat_features_xyz, axis=2)

            binary = helper_tf_util.conv2d(concat_features_xyz, 2, [1, 1], 'binary_pred', [1, 1], 'VALID', True, is_training, activation_fn=None)

            binary_soft = tf.nn.softmax(binary, axis=-1)
            binary_soft = tf.reshape(binary_soft, [batch_size, N, n_neighbors, 2])
            intra_neighbor = binary_soft[:, :, :, 1:]
            inter_neighbor = binary_soft[:, :, :, :1]

            # intra_similarity = tf.div_no_nan( # BxNxNeixC
            #     tf.reduce_sum(tf.multiply(features, neighbor_features), axis=-1, keep_dims=True),
            #     1e-10 + tf.multiply(tf.norm(features, axis=-1, keep_dims=True),
            #                         tf.norm(neighbor_features, axis=-1, keep_dims=True)))
            # intra_similarity = tf.div_no_nan(intra_similarity,
            #                                  1e-10 + tf.reduce_sum(intra_similarity, axis=-2, keep_dims=True))
            intra_features = tf.multiply(intra_neighbor, neighbor_features)
            intra_features = tf.div_no_nan(tf.reduce_sum(intra_features, axis=-2),
                                           tf.reduce_sum(intra_neighbor, axis=-2) + 1e-10)
            intra_features = tf.expand_dims(intra_features, axis=-2)
            concat_features_diff = tf.concat([features - neighbor_features, diff_xyz], axis=-1)
            concat_features_diff = tf.reshape(concat_features_diff, [batch_size, N, n_neighbors, input_dim+3])

            rel_feature = helper_tf_util.conv2d(concat_features_diff, fdim, [1, 1], 'rel_feature', [1, 1], 'VALID', True, is_training)

            inter_features = tf.multiply(inter_neighbor, rel_feature)
            inter_features = tf.div_no_nan(tf.reduce_sum(inter_features, axis=-2),
                                           tf.reduce_sum(inter_neighbor, axis=-2) + 1e-10)
            inter_features = tf.expand_dims(inter_features, axis=-2)

            net_intra = tf.concat([F, intra_features], axis=-1)
            net_intra = helper_tf_util.conv2d(net_intra, 32, [1, 1], 'intra_feature', [1, 1], 'VALID', True, is_training)
            net_inter = tf.concat([F, inter_features], axis=-1)
            net_inter = helper_tf_util.conv2d(net_inter, 32, [1, 1], 'inter_feature', [1, 1], 'VALID', True, is_training)
            cga_features = tf.concat([F, net_intra, net_inter], axis=-1)

        return cga_features, binary


    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits_cga,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0
        total_conf_matrix = np.zeros((self.config.num_classes, self.config.num_classes))

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                labels_tmp = np.arange(0, self.config.num_classes, 1)
                conf_matrix = confusion_matrix(
                                labels_valid, pred_valid, labels=labels_tmp)

                total_conf_matrix += conf_matrix
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list, precision_list, recall_list = [], [], []
        conf_matrix = total_conf_matrix
        for n in range(0, self.config.num_classes, 1):
            if float(gt_classes[n] + positive_classes[n] - true_positive_classes[n]) == 0:
                iou = 0
            else:
                iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)

            if float(positive_classes[n]) == 0:
                precision = 0
            else:
                precision = true_positive_classes[n] / float(positive_classes[n])
            precision_list.append(precision)

            if float(gt_classes[n]) == 0:
                recall = 0
            else: 
                recall = true_positive_classes[n] / float(gt_classes[n])
            recall_list.append(recall)

        overall_accuracy = val_total_correct / float(val_total_seen)
        mean_iou = sum(iou_list) / float(self.config.num_classes)
        mean_precision = sum(precision_list) / float(self.config.num_classes)
        mean_recall = sum(recall_list) / float(self.config.num_classes)

        if np.isnan(overall_accuracy):
            overall_accuracy = 0
        if np.isnan(mean_iou):
            mean_iou = 0
        if np.isnan(mean_precision):
            mean_precision = 0
        if np.isnan(mean_recall):
            mean_recall = 0

        # add scores to dictionary to log in tensorboard
        results_dict = {'OA': overall_accuracy, 'mIoU': mean_iou, 'mPrecision': mean_precision, 'mRecall': mean_recall}
        class_names = list(dataset.label_to_names.values())

        log_out('OA: {}'.format(overall_accuracy), self.Log_file)
        log_out('mean IoU: {}'.format(mean_iou), self.Log_file)
        log_out('mean precision: {}'.format(mean_precision), self.Log_file)
        log_out('mean recall: {}\n'.format(mean_recall), self.Log_file)

        # calculate iou per class
        log_out('IoU per class', self.Log_file)
        s = '{:5.2f} | '.format(100 * mean_iou)
        for class_name, IoU in zip(class_names, iou_list):
            if np.isnan(IoU):
                IoU = 0
            s += '{:5.2f} '.format(100 * IoU)
            results_dict['IoU ' + class_name] = IoU
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)

        # calculate precision per class
        log_out('Precision per class', self.Log_file)
        s = '{:5.2f} | '.format(100 * mean_precision)
        for class_name, precision in zip(class_names, precision_list):
            if np.isnan(precision):
                precision = 0
            s += '{:5.2f} '.format(100 * precision)
            results_dict['Precision ' + class_name] = precision
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)

        # calculate recall per class
        log_out('Recall per class', self.Log_file)
        s = '{:5.2f} | '.format(100 * mean_recall)
        for class_name, recall in zip(class_names, recall_list):
            if np.isnan(recall):
                recall = 0
            s += '{:5.2f} '.format(100 * recall)
            results_dict['Recall ' + class_name] = recall
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)

        if mean_iou > np.max(self.mIou_list):

            log_out('IoU List', self.Log_file)
            s = '['
            for i in iou_list:
                s += '{:5.2f}, '.format(100 * i)
            s = s[:-2]
            s += ']'
            log_out('{}\n'.format(s), self.Log_file)

            log_out('Precision List', self.Log_file)
            s = '['
            for i in precision_list:
                s += '{:5.2f}, '.format(100 * i)
            s = s[:-2]
            s += ']'
            log_out('{}\n'.format(s), self.Log_file)

            log_out('Recall List', self.Log_file)
            s = '['
            for i in recall_list:
                s += '{:5.2f}, '.format(100 * i)
            s = s[:-2]
            s += ']'
            log_out('{}\n'.format(s), self.Log_file)

            # log precision matrix as list
            log_out('Precision matrix as list:', self.Log_file)
            m = conf_matrix
            result = '['
            for i in range(len(m)):
                result = result + '['
                for j in range((len(m[i]))):
                    result = result + '{:0.3f}'.format(m[i][j]/sum(np.transpose(m)[j]))
                    if j < len(m[i]) - 1:
                        result = result + ', '
                if i < len(m) - 1:
                    result = result + '],\n'
                else:
                    result = result + ']]\n'
            log_out('{}'.format(result), self.Log_file)

            # log recall matrix as list
            log_out('Recall matrix as list:', self.Log_file)
            m = conf_matrix
            result = '['
            for i in range(len(m)):
                result = result + '['
                for j in range((len(m[i]))):
                    result = result + '{:0.3f}'.format(m[i][j]/sum(m[i]))
                    if j < len(m[i]) - 1:
                        result = result + ', '
                if i < len(m) - 1:
                    result = result + '],\n'
                else:
                    result = result + ']]\n'
            log_out('{}\n'.format(result), self.Log_file)

            # log confusion matrix
            log_out('Confusion matrix', self.Log_file)
            m = conf_matrix
            result = '['
            for i in range(len(m)):
                result = result + '['
                for j in range((len(m[i]))):
                    result = result + '{}'.format(int(m[i][j]))
                    if j < len(m[i]) - 1:
                        result = result + ', '
                if i < len(m) - 1:
                    result = result + '],\n'
                else:
                    result = result + ']]\n'
            log_out('{}'.format(result), self.Log_file)

        # log IoU, Precision and Recall scores for tensorboard
        for metric, score in results_dict.items():
            summary = tf.Summary()
            summary.value.add(tag=metric, simple_value = score)
            self.train_writer.add_summary(summary, self.training_epoch)

        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights=None):
        if pre_cal_weights is None:
            class_weights = tf.ones_like(logits, dtype=tf.float32)
        else:
            class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=tf.shape(logits)[-1])

        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def get_cb_focal_loss(self, logits, labels, pre_cal_weights, gamma=2.0, beta=0.9999999):
        # calculate class balanced focal loss
        epsilon = 1e-8
        model_out = tf.nn.softmax(logits)
        model_out = tf.clip_by_value(model_out, epsilon, 1-epsilon)
        onehot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        ce = tf.multiply(onehot_labels, -tf.log(model_out))
        effective_num = tf.subtract(1., tf.pow(beta, pre_cal_weights))
        weights = tf.divide(tf.subtract(1., beta), effective_num)
        weights = tf.multiply(tf.divide(weights, tf.reduce_sum(weights)), self.config.num_classes)
        weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(weights, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_mean(tf.reduce_sum(fl, axis=1))
        return reduced_fl

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg