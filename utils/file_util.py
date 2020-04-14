import os


def get_file_list(file_path):
    """
    获取文件夹内所有文件名
    :param file_path: 目标文件夹 string
    :return: 所有文件名 list()
    """
    file_list = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def create_dir(des_dir):
    """
    创建一个目录
    :param des_dir:
    :return:
    """
    if os.path.exists(des_dir):
        pass
    else:
        os.mkdir(des_dir)
