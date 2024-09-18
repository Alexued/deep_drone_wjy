import os
import pandas as pd
import time

import matplotlib.pyplot as plt

class DataCleaner:
    def __init__(self, csv_file_path, output_dir, picture_save_dir, sub_txt_file_path):
        self.csv_file_path = csv_file_path
        self.output_dir = output_dir
        self.picture_save_dir = picture_save_dir
        self.sub_txt_file_path = sub_txt_file_path
        self.get_data()


    def get_data(self):
        self.df = pd.read_csv(self.csv_file_path)
        self.rollout_idx = self.df['Rollout_idx'].values
        self.gt_p_x = self.df['gt_Position_x'].values
        self.gt_p_y = self.df['gt_Position_y'].values
        self.gt_p_z = self.df['gt_Position_z'].values
        self.gt_o_x = self.df['gt_Orientation_x'].values
        self.gt_o_y = self.df['gt_Orientation_y'].values
        self.gt_o_z = self.df['gt_Orientation_z'].values
        self.gt_o_w = self.df['gt_Orientation_w'].values
        self.ref_p_x = self.df['Reference_position_x'].values
        self.ref_p_y = self.df['Reference_position_y'].values
        self.ref_p_z = self.df['Reference_position_z'].values
        self.ref_o_x = self.df['Reference_orientation_x'].values
        self.ref_o_y = self.df['Reference_orientation_y'].values
        self.ref_o_z = self.df['Reference_orientation_z'].values
        self.ref_o_w = self.df['Reference_orientation_w'].values
        self.p_x = self.df['Position_x'].values
        self.p_y = self.df['Position_y'].values
        self.p_z = self.df['Position_z'].values
        self.o_x = self.df['Orientation_x'].values
        self.o_y = self.df['Orientation_y'].values
        self.o_z = self.df['Orientation_z'].values
        self.o_w = self.df['Orientation_w'].values

    def write_data_to_file(self, df_i, file_name, columns):
        output_path = os.path.join(self.output_dir, file_name)
        type = file_name.split('.')[-1]
        df_i[columns].to_csv(output_path, sep=' ', index=False, header=False)
        print(f"Data cleaning completed. Saved to {output_path}")

    def write_groundtruth_data(self, df_i):
        file_name = 'stamped_groundtruth.txt'
        columns = ['Odometry_stamp', 'Reference_position_x', 'Reference_position_y', 'Reference_position_z',
                   'Reference_orientation_x', 'Reference_orientation_y', 'Reference_orientation_z',
                   'Reference_orientation_w']
        self.write_data_to_file(df_i, file_name, columns)

    def write_estimate_data(self, df_i):
        file_name = 'stamped_traj_estimate.txt'
        columns = ['Odometry_stamp', 'gt_Position_x', 'gt_Position_y', 'gt_Position_z',
                   'gt_Orientation_x', 'gt_Orientation_y', 'gt_Orientation_z', 'gt_Orientation_w']
        self.write_data_to_file(df_i, file_name, columns)

    def write_full_data_to_xlsx(self, df_i, i, output_path):
        file_name = f'Rollout_idx_{i}.xlsx'
        columns = ['gt_Position_x', 'gt_Position_y', 'gt_Position_z',
                   'Reference_position_x', 'Reference_position_y', 'Reference_position_z',
                   'Position_x', 'Position_y', 'Position_z']
        output_path = os.path.join(output_path, file_name)
        df_i[columns].to_excel(output_path, index=False)
        print(f"Data cleaning completed. Saved to {output_path}")

    def clean_data(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        df = pd.read_csv(self.csv_file_path)

        idx = df['Rollout_idx'].values
        print(f"Total number of rollouts: {len(set(idx))}")
        for i in range(1, len(set(idx)) + 1):
            output_path = os.path.join(self.output_dir, f'Rollout_idx_{i}')
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)

            df_i = df[df['Rollout_idx'] == i]

            # self.write_groundtruth_data(df_i)
            # self.write_estimate_data(df_i)
            self.write_full_data_to_xlsx(df_i, i, output_path)

    def plot_single_picture(self, x, y, z, type):
        title = f"{type} plot"
        xlabel = f"{type}_x"
        ylabel = f"{type}_y"
        zlabel = f"{type}_z"

        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
        ax.scatter(x, y, z, color="red", label='gt_Position')
        plt.title(title)
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_zlabel(zlabel, fontweight='bold')
        plt.show()

    def plot_data(self, xlsx_file_path, picture_save_dir):
        file_name = os.path.basename(xlsx_file_path).split('.')[0]

        df = pd.read_excel(xlsx_file_path)
        gt_p_x = df['gt_Position_x'].values
        gt_p_y = df['gt_Position_y'].values
        gt_p_z = df['gt_Position_z'].values
        ref_p_x = df['Reference_position_x'].values
        ref_p_y = df['Reference_position_y'].values
        ref_p_z = df['Reference_position_z'].values
        p_x = df['Position_x'].values
        p_y = df['Position_y'].values
        p_z = df['Position_z'].values

        fig = plt.figure(figsize=(20, 20))
        ax = plt.axes(projection="3d")
        ax.scatter(gt_p_x, gt_p_y, gt_p_z, color="red", label='gt_Position', marker='^')
        ax.scatter(ref_p_x, ref_p_y, ref_p_z, color="blue", label='Reference_position', marker='o')
        ax.scatter(p_x, p_y, p_z, color="green", label='Position', marker='x')
        plt.title(file_name)
        ax.set_xlabel('x', fontweight='bold')
        ax.set_ylabel('y', fontweight='bold')
        ax.set_zlabel('z', fontweight='bold')
        plt.legend()  # Add legend
        plt.savefig(os.path.join(picture_save_dir, f"{file_name}.png"))

    def write_to_file(self, saved_path, df_i, columns):
        if os.path.exists(saved_path):
            os.remove(saved_path)
            # return
        timestamp = df_i[columns[0]].values
        tx = df_i[columns[1]].values
        ty = df_i[columns[2]].values
        tz = df_i[columns[3]].values
        qx = df_i[columns[4]].values
        qy = df_i[columns[5]].values
        qz = df_i[columns[6]].values
        qw = df_i[columns[7]].values

        with open(saved_path, 'w') as file:
            file.write("#timestamp tx ty tz qx qy qz qw\n")
            for i in range(len(timestamp)):
                file.write(f"{timestamp[i]} {tx[i]} {ty[i]} {tz[i]} {qx[i]} {qy[i]} {qz[i]} {qw[i]}\n")

    def parse_data(self):
        if not os.path.exists(self.sub_txt_file_path):
            os.makedirs(self.sub_txt_file_path, exist_ok=True)
        df = pd.read_csv(self.csv_file_path)

        for rollout_idx in set(self.rollout_idx):
            txt_file_path = os.path.join(self.sub_txt_file_path, f'Rollout_idx_{rollout_idx}')
            # print(f"txt_file_path: {txt_file_path}")
            if not os.path.exists(txt_file_path):
                os.makedirs(txt_file_path, exist_ok=True)
            gt_file_path = os.path.join(txt_file_path, 'stamped_groundtruth.txt')
            estimate_file_path = os.path.join(txt_file_path, 'stamped_traj_estimate.txt')
            # print(f"gt_file_path: {gt_file_path}")
            # print(f"estimate_file_path: {estimate_file_path}")
            df_i = self.df[self.df['Rollout_idx'] == rollout_idx]
            # print(f"df_i: {df_i}")
            gt_columns = ['Odometry_stamp', 'Reference_position_x', 'Reference_position_y', 'Reference_position_z',
                   'Reference_orientation_x', 'Reference_orientation_y', 'Reference_orientation_z',
                   'Reference_orientation_w']
            # print(f"len(df_i[gt_columns[0]]): {len(df_i[gt_columns[0]])}")
            # print(rollout_idx)
            # print(f"len(df_i[gt_columns[0]]): {len(df_i[gt_columns[0]])}")
            # print(f"gt_columns[0]: {gt_columns[0]}")
            # print(f"df_i[gt_columns[0]]: {df_i[gt_columns[0]]}")
            self.write_to_file(gt_file_path, df_i, gt_columns)
            estimate_columns = ['Odometry_stamp', 'gt_Position_x', 'gt_Position_y', 'gt_Position_z',
                   'gt_Orientation_x', 'gt_Orientation_y', 'gt_Orientation_z', 'gt_Orientation_w']
            self.write_to_file(estimate_file_path, df_i, estimate_columns)
            print(f"Rollout_idx_{rollout_idx} data cleaning completed. Saved to {gt_file_path} and {estimate_file_path}")
            time.sleep(0.5)
            # break


    def process_data(self, model='plot', save_picture=True):
        if model == 'write':
            self.clean_data()
        elif model == 'plot':
            xlsx_file_paths = [os.path.join(self.output_dir, folder) for folder in os.listdir(self.output_dir) if
                               os.path.isdir(os.path.join(self.output_dir, folder))]
            xlsx_file_paths.sort(key=lambda x: int(x.split('_')[-1]))

            if save_picture:
                # picture_save_dir = os.path.join(self.picture_save_dir, self.csv_file_path.split(os.sep)[-1].split('.')[0])
                if not os.path.exists(self.picture_save_dir):
                    os.makedirs(self.picture_save_dir, exist_ok=True)

                for xlsx_file_path in xlsx_file_paths:
                    xlsx_file_path = os.path.join(xlsx_file_path, os.listdir(xlsx_file_path)[0])
                    self.plot_data(xlsx_file_path, self.picture_save_dir)
                    print(f"Plotting {xlsx_file_path.split(os.sep)[-1]}, saving picture in {self.picture_save_dir}...")
                    time.sleep(0.5)
        elif model == 'txt':
            self.parse_data()
            # if not os.path.exists(self.sub_txt_file_path):
            #     os.makedirs(self.sub_txt_file_path, exist_ok=True)
            # for rollout_idx in set(self.rollout_idx):
            #     txt_file_paths = [os.path.join(self.sub_txt_file_path, f'Rollout_idx_{rollout_idx}')]
            #     # txt_file_paths.sort(key=lambda x: int(x.split('_')[-1]))
            #     print(f"txt_file_paths: {txt_file_paths}")


if __name__ == '__main__':
    csv_file_path = '/home/wjy/drone_acrobatics_ws/catkin_dda/src/deep_drone_acrobatics/controller_learning/data/loop/train'
    output_dir = '/home/wjy/drone_acrobatics_ws/catkin_dda/src/deep_drone_acrobatics/controller_learning/data/data_clean/xlsx_data'
    picture_save_dir = '/home/wjy/drone_acrobatics_ws/catkin_dda/src/deep_drone_acrobatics/controller_learning/data/data_clean/picture'
    txt_file_path = '/home/wjy/drone_acrobatics_ws/catkin_dda/src/deep_drone_acrobatics/controller_learning/data/data_clean/txt_data'

    # 这里的csv_files是一个列表，包含了csv_file_path目录下所有的.csv文件
    csv_files = [os.path.join(csv_file_path, file) for file in os.listdir(csv_file_path) if file.endswith('.csv')]
    # 这里的exist_files是一个列表，包含了output_dir目录下所有的文件夹
    exist_files = [folder for folder in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, folder))]
    
    for csv_file in csv_files:
        # if csv_file.split('/')[-1].split('.')[0] in exist_files:
        #     print(f'======={csv_file} already processed!========')
        #     continue
        # print(f'=======processing{csv_file}...========')
        sub_picture_save_dir = os.path.join(picture_save_dir, csv_file.split('/')[-1].split('.')[0])
        # print(f'sub_picture_save_dir: {sub_picture_save_dir}')
        sub_output_dir = os.path.join(output_dir, csv_file.split('/')[-1].split('.')[0])
        # print(f'sub_output_dir: {sub_output_dir}')
        sub_txt_file_path = os.path.join(txt_file_path, csv_file.split('/')[-1].split('.')[0])
        print(f'sub_txt_file_path: {sub_txt_file_path}')
        print(f'csv_file: {csv_file}')
        data_cleaner = DataCleaner(csv_file, sub_output_dir, sub_picture_save_dir, sub_txt_file_path)
        data_cleaner.process_data(model='txt', save_picture=False)
        # print(f'======={csv_file} write completed!========')
        # data_cleaner.process_data(model='plot', save_picture=True)
        # print(f'======={csv_file} process completed!========')