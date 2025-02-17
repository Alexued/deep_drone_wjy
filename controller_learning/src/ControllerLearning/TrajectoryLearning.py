#!/usr/bin/env python3
import csv
import datetime
import os
import random

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand
from std_msgs.msg import Bool
from std_msgs.msg import Empty

from .TrajectoryBase import TrajectoryBase, TRACK_NUM_NORMALIZE
import time


class TrajectoryLearning(TrajectoryBase):
    def __init__(self, config, mode):
        super(TrajectoryLearning, self).__init__(config, mode)
        self.gt_odometry = Odometry()
        self.latest_thrust_factor = 1.0 # Init, can be changed for different drones
        self.recorded_samples = 0
        if self.mode == 'training':
            return  # Nothing to initialize
        self.pub_reset_vio = rospy.Publisher("/vins_restart",
                                             Bool, queue_size=1)
        self.success_subs = rospy.Subscriber("success_reset", Empty,
                                             self.callback_success_reset,
                                             queue_size=1)
        # 接受vio的数据
        self.state_estimate_sub = rospy.Subscriber("/hummingbird/odometry_converted_vio",
                                                   Odometry,
                                                   self.callback_odometry,
                                                   queue_size=1,
                                                   tcp_nodelay=True)
        # 接受gt真实数据
        self.ground_truth_odom = rospy.Subscriber("/hummingbird/ground_truth/odometry",
                                                  Odometry,
                                                  self.callback_gt_odometry,
                                                  queue_size=1)
        # 接受vins估计的imu融合数据
        self.vins_mono_sub = rospy.Subscriber("/vins_estimator/imu_propagate", Odometry,
                                              self.callback_vins_mono, queue_size=1)
        self.infer_time = []
        # print('TrajectoryLearning init done')
        # 如果是iterative迭代模式，就开始写入csv文件
        if mode == "iterative" or self.config.verbose:
            self.write_csv_header()
        if self.mode == "testing":
            self.success = 1

    def train(self):
        self.is_training = True
        self.learner.train()
        self.is_training = False
        self.use_network = False

    def callback_success_reset(self, data):
        print("Received call to Clear Buffer and Restart Experiment")
        os.system("rosservice call /gazebo/pause_physics")
        self.rollout_idx += 1
        self.reset_queue()
        print('Buffer Cleared')
        if self.mode == 'testing':
            self.success = 1  # We are positive, default is pass
        # Init is hacky, but gazebo is very bad!
        self.n_times_expert = 0.000
        self.n_times_net = 0.001
        self.reference_updated = False
        os.system("rosservice call /gazebo/unpause_physics")
        print('Done Reset')

    def callback_gt_odometry(self, data):
        self.gt_odometry = data

    def callback_vins_mono(self, data):
        # print('===in callback_vins_mono===')
        self.vins_odometry = data
        # print(f"vins_odometry: {self.vins_odometry}")

    @property
    def vio_init_good(self):
        if self.vins_odometry is None:
            rospy.logwarn('vins_odometry is None, restart VIO and sleep 2s...')
            # os.system("timeout 1s rostopic pub /vins_restart std_msgs/Bool 'data: true'  ")
            # time.sleep(2)
            return 3
        max_allowed_velocity = 0.1
        if abs(self.vins_odometry.twist.twist.linear.x) < max_allowed_velocity and \
                abs(self.vins_odometry.twist.twist.linear.y) < max_allowed_velocity and \
                abs(self.vins_odometry.twist.twist.linear.z) < max_allowed_velocity:
            rospy.loginfo(f'VIO init is GOOD ,now {abs(self.vins_odometry.twist.twist.linear.x)},'
                  f'{abs(self.vins_odometry.twist.twist.linear.y)},'
                  f'{abs(self.vins_odometry.twist.twist.linear.z)}')
            return True
        else:
            rospy.logwarn(f'VIO init is BAD ,now {self.vins_odometry.twist.twist.linear.x},'
                  f'{self.vins_odometry.twist.twist.linear.y},'
                  f'{self.vins_odometry.twist.twist.linear.z}')
            return False

    def save_data(self):
        row = [self.rollout_idx,
               self.odometry.header.stamp.to_sec(),
               # GT Positon
               self.gt_odometry.pose.pose.position.x,
               self.gt_odometry.pose.pose.position.y,
               self.gt_odometry.pose.pose.position.z,
               self.ref_state.pose.position.z - self.gt_odometry.pose.pose.position.z,
               self.gt_odometry.pose.pose.orientation.w,
               self.gt_odometry.pose.pose.orientation.x,
               self.gt_odometry.pose.pose.orientation.y,
               self.gt_odometry.pose.pose.orientation.z,
               self.gt_odometry.twist.twist.linear.x,
               self.gt_odometry.twist.twist.linear.y,
               self.gt_odometry.twist.twist.linear.z,
               self.gt_odometry.twist.twist.angular.x,
               self.gt_odometry.twist.twist.angular.y,
               self.gt_odometry.twist.twist.angular.z,
               # VIO Estimate
               self.odometry.pose.pose.position.x,
               self.odometry.pose.pose.position.y,
               self.odometry.pose.pose.position.z,
               self.ref_state.pose.position.z - self.odometry.pose.pose.position.z,
               self.odometry.pose.pose.orientation.w,
               self.odometry.pose.pose.orientation.x,
               self.odometry.pose.pose.orientation.y,
               self.odometry.pose.pose.orientation.z,
               self.odometry.twist.twist.linear.x,
               self.odometry.twist.twist.linear.y,
               self.odometry.twist.twist.linear.z,
               self.odometry.twist.twist.angular.x,
               self.odometry.twist.twist.angular.y,
               self.odometry.twist.twist.angular.z,
               # Reference state
               self.ref_state.pose.position.x,
               self.ref_state.pose.position.y,
               self.ref_state.pose.position.z,
               self.ref_state.pose.orientation.w,
               self.ref_state.pose.orientation.x,
               self.ref_state.pose.orientation.y,
               self.ref_state.pose.orientation.z,
               self.ref_state.velocity.linear.x,
               self.ref_state.velocity.linear.y,
               self.ref_state.velocity.linear.z,
               self.ref_state.velocity.angular.x,
               self.ref_state.velocity.angular.y,
               self.ref_state.velocity.angular.z,
               # MPC output with GT Position
               self.control_command.collective_thrust,
               self.control_command.bodyrates.x,
               self.control_command.bodyrates.y,
               self.control_command.bodyrates.z,
               # NET output with GT Position
               self.net_control.collective_thrust,
               self.net_control.bodyrates.x,
               self.net_control.bodyrates.y,
               self.net_control.bodyrates.z,
               # Maneuver type
               0]

        if self.record_data and self.gt_odometry.pose.pose.position.z > 0.3 and \
                self.control_command.collective_thrust > 0.2:
            with open(self.csv_filename, 'a') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows([row])
            fts_name = '{:08d}.npy'
            fts_filename = os.path.join(self.image_save_dir,
                                        fts_name.format(self.recorded_samples))
            print(f"in save_data, self.features: {self.features}")
            np.save(fts_filename, self.features)
            self.recorded_samples += 1

    def callback_control_command(self, data):
        # print('===in callback_control_command===')
        self.control_command = data
        # print(f"control_command: {self.control_command}")
        self._generate_control_command()
        if self.mode == 'testing' and self.gt_odometry.pose.pose.position.z < 0.3:
            self.success = 0

    def compute_trajectory_error(self):
        gt_ref = np.array([self.ref_state.pose.position.x,
                           self.ref_state.pose.position.y,
                           self.ref_state.pose.position.z])
        gt_pos = np.array([self.gt_odometry.pose.pose.position.x,
                           self.gt_odometry.pose.pose.position.y,
                           self.gt_odometry.pose.pose.position.z])
        results = {"gt_ref": gt_ref, "gt_pos": gt_pos}
        return results

    def publish_control_command(self, control_command):
        # print('===in publish_control_command===')
        control_command.collective_thrust = self.latest_thrust_factor * control_command.collective_thrust
        # print(f'control_command.collective_thrust: {control_command.collective_thrust}')
        # 给/control_command话题发布油门数据，但是不知道发了多少次
        self.pub_actions.publish(control_command)

    def _generate_control_command(self):
        # print('===in _generate_control_command===')
        inputs = self._prepare_net_inputs()
        if not self.net_initialized:
            # Apply Network to init
            print("===init network===")
            results = self.learner.inference(inputs)
            print("Net initialized")
            self.net_initialized = True
            self.publish_control_command(self.control_command)

        if not self.use_network or not self.reference_updated \
                or (len(inputs['fts'].shape) != 4):
            # Will be in here if:
            # - starting and VIO init
            # - Image queue is not ready, can only run expert
            self.publish_control_command(self.control_command)
            if self.use_network:
                print("Using expert wait for ref")
            return
        # Use always expert at the beginning (approximately 0.2s) to avoid syncronization problems
        if self.counter < 10:
            self.counter += 1
            self.publish_control_command(self.control_command)
            return

        # Apply Network
        # 这里产生了一次网络的推理
        # print('===before inference in _generate_control_command===')
        # start_time = datetime.datetime.now()
        results = self.learner.inference(inputs)
        end_time = datetime.datetime.now()
        # inference_time = end_time.microsecond - start_time.microsecond
        inference_time = 0
        # print(f"====Inference time: {inference_time}====")
        if inference_time <= 10000 and inference_time >= 0:
            self.infer_time.append(abs(inference_time))
        # print('===after inference in _generate_control_command===')
        control_command = ControlCommand()
        control_command.armed = True
        control_command.expected_execution_time = rospy.Time.now()
        control_command.control_mode = 2
        control_command.collective_thrust = results[0][0].numpy()
        control_command.bodyrates.x = results[0][1].numpy()
        control_command.bodyrates.y = results[0][2].numpy()
        control_command.bodyrates.z = results[0][3].numpy()
        self.net_control = control_command
        # print(f'self.net_control: {self.net_control}')
        
        print(f"in _generate_control_command, self.features: {self.features}")
        # Log immediately everything to avoid surprises (if required)
        if self.record_data:
            self.save_data()

        # Apply random controller now and then to facilitate exploration
        # 不是在测试模式下进行随即加偏差
        if (self.mode != 'testing') and random.random() < self.config.rand_controller_prob:
            self.control_command.collective_thrust += self.config.rand_thrust_mag * (random.random() - 0.5) * 2
            self.control_command.bodyrates.x += self.config.rand_rate_mag * (random.random() - 0.5) * 2
            self.control_command.bodyrates.y += self.config.rand_rate_mag * (random.random() - 0.5) * 2
            self.control_command.bodyrates.z += self.config.rand_rate_mag * (random.random() - 0.5) * 2
            self.publish_control_command(self.control_command)
            return

        # Dagger (on control command label).
        # 不知道这里的d是给谁的，为什么要减掉control_command，为什么这样会成为后面的判断条件
        # 这里应该有个判断条件，如果d_thrust, d_br_x, d_br_y, d_br_z都小于某个阈值，就用网络的输出，否则用专家的输出
        # 可能是上一个control_command和这一个control_command的差值，判断稳定性？
        # 看着像和mpc的对比
        d_thrust = control_command.collective_thrust - self.control_command.collective_thrust
        d_br_x = control_command.bodyrates.x - self.control_command.bodyrates.x
        d_br_y = control_command.bodyrates.y - self.control_command.bodyrates.y
        d_br_z = control_command.bodyrates.z - self.control_command.bodyrates.z
        if self.config.execute_nw_predictions \
                and abs(d_thrust) < self.config.fallback_threshold_rates \
                and abs(d_br_x) < self.config.fallback_threshold_rates \
                and abs(d_br_y) < self.config.fallback_threshold_rates \
                and abs(d_br_z) < self.config.fallback_threshold_rates:
            self.n_times_net += 1
            self.publish_control_command(control_command)
            print(f"self.n_times_net: {self.n_times_net}")
        else:
            self.n_times_expert += 1
            self.publish_control_command(self.control_command)

    def write_csv_header(self):
        # row是csv文件的第一行，包含GT Position, VIO Estimate, Reference state, MPC output with GT Postion, Net output
        row = ["Rollout_idx",
               "Odometry_stamp",
               # GT Position
               "gt_Position_x",
               "gt_Position_y",
               "gt_Position_z",
               "gt_Position_z_error",
               "gt_Orientation_w",
               "gt_Orientation_x",
               "gt_Orientation_y",
               "gt_Orientation_z",
               "gt_V_linear_x",
               "gt_V_linear_y",
               "gt_V_linear_z",
               "gt_V_angular_x",
               "gt_V_angular_y",
               "gt_V_angular_z",
               # VIO Estimate
               "Position_x",
               "Position_y",
               "Position_z",
               "Position_z_error",
               "Orientation_w",
               "Orientation_x",
               "Orientation_y",
               "Orientation_z",
               "V_linear_x",
               "V_linear_y",
               "V_linear_z",
               "V_angular_x",
               "V_angular_y",
               "V_angular_z",
               # Reference state
               "Reference_position_x",
               "Reference_position_y",
               "Reference_position_z",
               "Reference_orientation_w",
               "Reference_orientation_x",
               "Reference_orientation_y",
               "Reference_orientation_z",
               "Reference_v_linear_x",
               "Reference_v_linear_y",
               "Reference_v_linear_z",
               "Reference_v_angular_x",
               "Reference_v_angular_y",
               "Reference_v_angular_z",
               # MPC output with GT Postion
               "Gt_control_command_collective_thrust",
               "Gt_control_command_bodyrates_x",
               "Gt_control_command_bodyrates_y",
               "Gt_control_command_bodyrates_z",
               # Net output
               "Net_control_command_collective_thrust",
               "Net_control_command_bodyrates_x",
               "Net_control_command_bodyrates_y",
               "Net_control_command_bodyrates_z",
               "Maneuver_type"]

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.mode == 'iterative':
            root_save_dir = self.config.train_dir
        else:
            root_save_dir = self.config.log_dir
        self.csv_filename = os.path.join(root_save_dir, "data_" + current_time + ".csv")
        self.image_save_dir = os.path.join(root_save_dir, "img_data_" + current_time)
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)
        with open(self.csv_filename, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows([row])
