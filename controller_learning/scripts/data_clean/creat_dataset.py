import os
import numpy as np
import pandas as pd
from datetime import datetime

def get_npy(dataset_path):
    file_names = os.listdir(dataset_path)
    for file_name in file_names:
        if not file_name.endswith('.csv'):
            npy_file = os.path.join(dataset_path, file_name)
    npy_filename = os.listdir(npy_file)
    npy_files = []
    for npy_file in npy_filename:
        if npy_file.endswith('.npy'):
            npy_files.append(npy_file)
    sorted_npy_files = sorted(npy_files, key=lambda x: int(x.split('.')[0]))
    return sorted_npy_files

def get_csv(dataset_path):
    file_names = os.listdir(dataset_path)
    for file_name in file_names:
        if file_name.endswith('.csv'):
            csv_file = os.path.join(dataset_path, file_name)
    return csv_file

def get_dataset(npy_files, csv_file, percentage):
    dataset = []

    df = pd.read_csv(csv_file)

    idx = len(df)
    # print(idx)
    num_nup = len(npy_files)
    if idx == num_nup:
        print('csv行数和npy文件数量一致, 可以继续处理')
    else:
        print('csv行数和npy文件数量不一致, 请检查')
        return dataset
    assert idx == num_nup

    assert percentage > 0 and percentage < 1
    train_num = int(idx * percentage)
    test_num = idx - train_num
    assert train_num + test_num == idx
    train_data = []
    test_data = []
    for train_idx in range(idx):
        if train_idx < train_num:
            train_data.append([npy_files[train_idx], df.iloc[train_idx, :].values])
        else:
            test_data.append([npy_files[train_idx], df.iloc[train_idx, :].values])
    dataset = [train_data, test_data]
    return dataset

def create_dataset(dataset_path, train_output_path, test_output_path, dataset): 
    sub_path = os.listdir(dataset_path)
    for path in sub_path:
        if path.endswith('.csv'):
            # csv_file_path = os.path.join(dataset_path, path)
            csv_name = path
        else:
            npy_file_name = path
            npy_file_path = os.path.join(dataset_path, path)
    train_npy_output_path = os.path.join(train_output_path, npy_file_name)
    test_npy_output_path = os.path.join(test_output_path, npy_file_name)
    if not os.path.exists(train_npy_output_path):
        os.makedirs(train_npy_output_path)
    if not os.path.exists(test_npy_output_path):
        os.makedirs(test_npy_output_path)
    train_data = dataset[0]
    test_data = dataset[1]
    full_csv_data = []
    for train in train_data:
        npy_file = train[0]
        csv_data = train[1]
        # 将npy文件复制到指定目录
        os.system('cp ' + os.path.join(npy_file_path, npy_file) + ' ' + os.path.join(train_npy_output_path, npy_file))
        # 验证文件是否复制成功
        if os.path.exists(os.path.join(train_npy_output_path, npy_file)):
            print(f'npy file {npy_file} has been copied to {train_npy_output_path}')
        else:
            print(f'npy file {npy_file} has not been copied to {train_npy_output_path}')
        # 将csv数据写入到指定文件
        full_csv_data.append(csv_data)
    column_names = [
    "Rollout_idx", "Odometry_stamp", "gt_Position_x", "gt_Position_y", "gt_Position_z",
    "gt_Position_z_error", "gt_Orientation_w", "gt_Orientation_x", "gt_Orientation_y",
    "gt_Orientation_z", "gt_V_linear_x", "gt_V_linear_y", "gt_V_linear_z",
    "gt_V_angular_x", "gt_V_angular_y", "gt_V_angular_z", "Position_x", "Position_y",
    "Position_z", "Position_z_error", "Orientation_w", "Orientation_x", "Orientation_y",
    "Orientation_z", "V_linear_x", "V_linear_y", "V_linear_z", "V_angular_x",
    "V_angular_y", "V_angular_z", "Reference_position_x", "Reference_position_y",
    "Reference_position_z", "Reference_orientation_w", "Reference_orientation_x",
    "Reference_orientation_y", "Reference_orientation_z", "Reference_v_linear_x",
    "Reference_v_linear_y", "Reference_v_linear_z", "Reference_v_angular_x",
    "Reference_v_angular_y", "Reference_v_angular_z", "Gt_control_command_collective_thrust",
    "Gt_control_command_bodyrates_x", "Gt_control_command_bodyrates_y",
    "Gt_control_command_bodyrates_z", "Net_control_command_collective_thrust",
    "Net_control_command_bodyrates_x", "Net_control_command_bodyrates_y",
    "Net_control_command_bodyrates_z", "Maneuver_type"
    ]
    full_csv_data = np.array(full_csv_data)
    full_csv_data = pd.DataFrame(full_csv_data, columns=column_names)
    full_csv_data.to_csv(os.path.join(train_output_path, csv_name), index=False)
    if os.path.exists(os.path.join(train_output_path, csv_name)):
        print(f'csv file {csv_name} has been written to {train_output_path}')
    else:
        print(f'csv file {csv_name} has not been written to {train_output_path}')
    print('train data has been processed')

    full_csv_data = []
    for idx, test in enumerate(test_data):
        npy_file = test[0]
        csv_data = test[1]
        # 将npy文件复制到指定目录, 复制后的文件名为00000000.npy这样的格式
        os.system('cp ' + os.path.join(npy_file_path, npy_file) + ' ' + os.path.join(test_npy_output_path, str(idx).zfill(8) + '.npy'))
        if os.path.exists(os.path.join(test_npy_output_path, str(idx).zfill(8) + '.npy')):
            print(f'npy file {npy_file} has been copied to {test_npy_output_path}')
        # 将csv数据写入到指定文件
        full_csv_data.append(csv_data)
    full_csv_data = np.array(full_csv_data)
    full_csv_data = pd.DataFrame(full_csv_data, columns=column_names)
    full_csv_data.to_csv(os.path.join(test_output_path, csv_name), index=False)
    if os.path.exists(os.path.join(test_output_path, csv_name)):
        print(f'csv file {csv_name} has been written to {test_output_path}')
    else:
        print(f'csv file {csv_name} has not been written to {test_output_path}')
    print('test data has been processed')

def exchange_dataset(train_output_path, test_output_path, loop_path, current_time):
    
    train_dir = os.path.join(loop_path, 'train')
    test_dir = os.path.join(loop_path, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    # 将train和test目录下的文件移动到train_bak和test_bak下
    os.system('mv ' + train_dir + ' ' + loop_path + '/train_bak_' + current_time)
    os.system('mv ' + test_dir + ' ' + loop_path + '/test_bak_' + current_time)
    if os.path.exists(loop_path + '/train_bak_' + current_time) and os.path.exists(loop_path + '/test_bak_' + current_time):
        print('train and test have been moved to train_bak and test_bak')
    else:
        print('train and test have not been moved to train_bak and test_bak')
    # 将train_output_path和test_output_path里的文件复制到loop_path下
    os.system('cp -r ' + train_output_path + ' ' + loop_path)
    os.system('cp -r ' + test_output_path + ' ' + loop_path)
    if os.path.exists(loop_path + f'/cleaned_train_{current_time}') and os.path.exists(loop_path + f'/cleaned_test_{current_time}'):
        print('train_output_path and test_output_path have been copied to loop')
    else:
        print('train_output_path and test_output_path have not been copied to loop')
    # 将名字改为train和test
    os.system('mv ' + loop_path + f'/cleaned_train_{current_time}' + ' ' + train_dir)
    os.system('mv ' + loop_path + f'/cleaned_test_{current_time}' + ' ' + test_dir)
    print('train_output_path and test_output_path have been copied to loop')

def get_dataset_path(base_dir):
    sub_dirs = os.listdir(base_dir)
    # check if there is csv file in the directory,如果目录下没有csv文件，表示之前已经处理过，直接返回
    has_csv = any(file.endswith('.csv') for file in sub_dirs)
    if not has_csv:
        print('no csv file in the directory, return')
        return sub_dirs
    else:
        print('csv file in the directory, continue')
    csv_names = []
    dataset_path = []
    paired_path = []
    new_floders = []
    if len(sub_dirs) == 1:
        dataset_path = os.path.join(base_dir, sub_dirs[0])
    else:
        for sub_dir in sub_dirs:
            if sub_dir.endswith('.csv'):
                csv_names.append(sub_dir.split('.')[0])
                new_floder = os.path.join(base_dir, sub_dir.split('.')[0])
                new_floders.append(new_floder)
                
        for sub_dir in sub_dirs:
            if sub_dir.endswith('.csv'):
                continue
            for csv_name in csv_names:
                if csv_name in sub_dir:
                    dataset_path.append(os.path.join(base_dir, sub_dir))
    for path in dataset_path:
        for csv_name in csv_names:
            if csv_name in path:
                paired_path.append([path, csv_name + '.csv'])
    # after getting the dataset_path, csv_name and new_floders, create new floders
    for new_floder in new_floders:
        if not os.path.exists(new_floder):
            os.makedirs(new_floder)
            print(f'{new_floder} has been created')
    # sort the paired_path and new_floders
    paired_path = sorted(paired_path, key=lambda x: x[1])
    new_floders = sorted(new_floders)
    # move paired to new folder
    print(f'paired_path: {paired_path}')
    print(f'len(paired_path): {len(paired_path)}')
    paired_num = int(len(paired_path))
    print(f'paired_num: {paired_num}')
    for i in range(paired_num):
        print(f'i={i}')
        os.system('mv ' + paired_path[i][0] + ' ' + new_floders[i])
        if paired_path[i][0] in os.listdir(new_floders[i]):
            print(f'{paired_path[i][0]} has been moved to {new_floders[i]}')
        os.system('mv ' + os.path.join(base_dir, paired_path[i][1]) + ' ' + new_floders[i])
        if paired_path[i][1] in os.listdir(new_floders[i]):
            print(f'{paired_path[i][1]} has been moved to {new_floders[i]}')
    # return os.listdir(base_dir)
    return new_floders


if __name__ == '__main__':
    base_dir = '/home/wjy/drone_acrobatics_ws/catkin_dda/src/deep_drone_acrobatics/controller_learning/data/loop/train_bak_0831'
    current_time = datetime.now().strftime('%m%d%H%M%S')
    percentage = 0.8 # 训练集占比
    # dataset_path = '/home/wjy/drone_acrobatics_ws/catkin_dda/src/deep_drone_acrobatics/controller_learning/data/loop/test'
    dataset_paths = get_dataset_path(base_dir)
    
    for dataset_path in dataset_paths:
        print(dataset_path)

        # 获取npy文件和csv文件
        npy_files = get_npy(dataset_path)
        csv_file = get_csv(dataset_path)

        # 获取数据集
        dataset = get_dataset(npy_files, csv_file, percentage)
        if dataset == []:
            print('数据集分割失败，请检查文件数量是否对上!')
        
        # 创建数据集
        output_path = '/home/wjy/drone_acrobatics_ws/catkin_dda/src/deep_drone_acrobatics/controller_learning/data/loop/cleaned_dataset'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        train_output_path = os.path.join(output_path, f'cleaned_train_{current_time}')
        test_output_path = os.path.join(output_path, f'cleaned_test_{current_time}')
        if not os.path.exists(train_output_path) and not os.path.exists(test_output_path):
            os.makedirs(train_output_path)
            os.makedirs(test_output_path)
        create_dataset(dataset_path, train_output_path, test_output_path, dataset)
        
        # 将train和test复制到loop目录下
        loop_path = '/'.join(dataset_path.split('/')[:-1])
        exchange_dataset(train_output_path, test_output_path, loop_path, current_time)
        print('数据集处理完成')