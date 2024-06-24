import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

# Required for catkin import strategy
try:
    from .nets import create_network
    from .body_dataset import create_dataset
except:
    from nets import create_network
    from body_dataset import create_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

class BodyrateLearner(object):
    def __init__(self, settings):
        self.config = settings # 读取配置文件，赋值到self.config
        # 检测设备，读取GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print(f"======USE DEVICE: {physical_devices[0].name}======")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.min_val_loss = tf.Variable(np.inf,
                                        name='min_val_loss',
                                        trainable=False)

        self.network = create_network(self.config)
        # print struct of model
        # states_conv_model = self.network.states_conv
        # print('--------model struct--------')
        # self.network.build((self.config.seq_len, int(64*2)))
        # self.network.summary()
        # print('--------model struct--------')

        self.loss = tf.keras.losses.MeanSquaredError() # 定义损失函数为均方差损失函数
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=.2)

        # 计算给定值的（加权）平均值
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='validation_loss')

        self.global_epoch = tf.Variable(0)

        self.ckpt = tf.train.Checkpoint(step=self.global_epoch,
                                        optimizer=self.optimizer,
                                        net=self.network)

        if self.config.resume_training:
            if self.ckpt.restore(self.config.resume_ckpt_file):
                print("------------------------------------------")
                print("Restored from {}".format(self.config.resume_ckpt_file))
                print("------------------------------------------")
                return

        print("------------------------------------------")
        print("Initializing from scratch.")
        print("------------------------------------------")

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.network(inputs)
            # print('--------model struct--------')
            self.network.summary()
            plot_model(self.network, to_file='mdoel_struct.png', show_shapes=True)
            # layer = self.network.get_layer(name='temporal_conv_net_1')
            # print(layer.get_config())
            # print('--------model struct--------')
            loss = self.loss(labels, predictions)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        self.train_loss.update_state(loss)
        return gradients

    @tf.function
    def val_step(self, inputs, labels):
        predictions = self.network(inputs)
        loss = self.loss(labels, predictions)
        self.val_loss.update_state(loss)

    def adapt_input_data(self, features):
        if self.config.use_fts_tracks:
            inputs = {"fts": features[1],
                      "state": features[0]}
        else:
            inputs = {"state": features}
        return inputs

    def write_train_summaries(self, features, gradients):
        with self.summary_writer.as_default():
            tf.summary.scalar('Train Loss', self.train_loss.result(),
                              step=self.optimizer.iterations)
            # add graph
            # tf.summary.graph(tf.compat.v1.get_default_graph())
            # tf.summary.trace_on(graph=True, profiler=True)
            for g, v in zip(gradients, self.network.trainable_variables):
                tf.summary.histogram(v.name, g, step=self.optimizer.iterations)
        tf.summary.trace_off()

    def train(self):
        print("Training Network")
        if not hasattr(self, 'train_log_dir'):
            print('======hasattr is called======')
            # This should be done only once
            self.train_log_dir = os.path.join(self.config.log_dir, 'train')
            print(f'self.train_log_dir: {self.train_log_dir}')
            self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.train_log_dir, max_to_keep=10)
        else:
            # We are in dagger mode, so let us reset the best loss
            print('We are in dagger mode, so let us reset the best loss')
            self.min_val_loss = np.inf
            self.train_loss.reset_states()
            self.val_loss.reset_states()

        print(f"config.train_dir: {self.config.train_dir}")
        dataset_train = create_dataset(self.config.train_dir,
                                       self.config, training=True)
        print(f'======dataset_train is set, next set dataset_val========')
        print(f"config.val_dir: {self.config.val_dir}")
        dataset_val = create_dataset(self.config.val_dir,
                                     self.config, training=False)

        for epoch in range(self.config.max_training_epochs):
            # Train
            for k, (features, label) in enumerate(tqdm(dataset_train.batched_dataset)):
                features = self.adapt_input_data(features)
                gradients = self.train_step(features, label)
                if tf.equal(k % self.config.summary_freq, 0):
                    self.write_train_summaries(features, gradients)
                    self.train_loss.reset_states()
            # Eval
            for features, label in tqdm(dataset_val.batched_dataset):
                features = self.adapt_input_data(features)
                self.val_step(features, label)
            validation_loss = self.val_loss.result()
            with self.summary_writer.as_default():
                # tf.summary.trace_on(graph=True, profiler=True)
                tf.summary.scalar("Validation Loss", validation_loss, step=tf.cast(self.global_epoch, dtype=tf.int64))
            # tf.summary.trace_off()
            self.val_loss.reset_states()

            self.global_epoch = self.global_epoch + 1
            self.ckpt.step.assign_add(1)
            print(f"========ues train_dir is {self.config.train_dir}========")
            print("Epoch: {:2d}, Validation Loss: {:.4f}".format(self.global_epoch, validation_loss))

            if validation_loss < self.min_val_loss or ((epoch + 1) % self.config.save_every_n_epochs) == 0:
                if validation_loss < self.min_val_loss:
                    self.min_val_loss = validation_loss
                save_path = self.ckpt_manager.save()
                print("Saved checkpoint for epoch {}: {}".format(int(self.ckpt.step), save_path))

        # Reset the metrics for the next epoch
        print("------------------------------")
        print("Training finished successfully")
        print("------------------------------")

    def test(self):
        print("Testing Network")
        self.train_log_dir = os.path.join(self.config.log_dir, 'test')
        dataset_val = create_dataset(self.config.test_dir,
                                     self.config, training=False)

        for features, label in tqdm(dataset_val.batched_dataset):
            features = self.adapt_input_data(features)
            self.val_step(features, label)
        validation_loss = self.val_loss.result()
        self.val_loss.reset_states()

        print("Testing Loss: {:.4f}".format(validation_loss))

    @tf.function
    def inference(self, inputs):
        predictions = self.network(inputs)
        return predictions
