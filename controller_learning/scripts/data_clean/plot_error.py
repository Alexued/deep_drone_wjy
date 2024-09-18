# -*- coding: utf-8 -*-
import os
import subprocess
import time
import re

# 设定你的目录路径
base_dir = '/home/wjy/Trajectory_evaluation/data_20240901-005242'

# 获取目录下的所有子目录，并根据其中的数字进行排序
def get_sorted_subdirectories(base_dir):
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]
    
    # 提取子目录中的数字，并进行排序
    def extract_number(dir_name):
        match = re.search(r'(\d+)', dir_name)
        return int(match.group(1)) if match else 0

    sorted_subdirs = sorted(subdirs, key=lambda d: extract_number(os.path.basename(d)))
    return sorted_subdirs

# 执行命令
def execute_command(path):
    cmd = ['python2', '/home/wjy/catkin_ws/src/rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py', path, '--png', '--plot']
    subprocess.call(cmd)

def main():
    paths = get_sorted_subdirectories(base_dir)
    total_count = len(paths)
    
    for idx, path in enumerate(paths):
        # 检查是否有名为 'plots' 的子目录
        if 'plots' in os.listdir(path) and os.path.isdir(os.path.join(path, 'plots')):
            print("跳过目录: {} (已存在 'plots' 子目录)".format(path))
            continue
        execute_command(path)
        remaining_count = total_count - (idx + 1)
        print("==================================================处理完{}, 已处理完 {} 个子目录，还剩余 {} 个子目录未处理。==================================================".format(path.split('/')[-1], idx + 1, remaining_count))
        time.sleep(0.5)

if __name__ == '__main__':
    main()
