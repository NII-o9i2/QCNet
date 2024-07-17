import os
import shutil
from math import ceil

def split_folder_into_subfolders(folder_path, num_subfolders):
    # 获取文件夹中的所有文件夹
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    total_folders = len(folders)

    # 计算每个子文件夹应该包含的文件夹数量
    folders_per_sub_folder = ceil(total_folders / num_subfolders)
    print('Total folders: {}'.format(total_folders), 'sub_folders: {}'.format(folders_per_sub_folder))

    # # 创建子文件夹并分配文件夹
    for i in range(num_subfolders):
        sub_folder_name = os.path.join(folder_path, f"train_part_{i + 1}")
        os.makedirs(sub_folder_name, exist_ok=True)
        print('Process sub folder: {}'.format(sub_folder_name))
        # 分配文件夹到子文件夹
        moved_count = 0
        for j in range(i * folders_per_sub_folder, min((i + 1) * folders_per_sub_folder, total_folders)):
            shutil.move(os.path.join(folder_path, folders[j]), sub_folder_name)
            moved_count = moved_count + 1

            # 每移动 1000 个文件夹打印一次进度
            if moved_count % 1000 == 0 or moved_count == total_folders:
                print(f"Moved {moved_count} of {folders_per_sub_folder} folders")

folder_path = '/data/argoverse_data/v2/train/'  # 替换为你的 train 文件夹路径
num_sub_folders = 10
split_folder_into_subfolders(folder_path, num_sub_folders)
