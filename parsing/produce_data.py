from shift_reduce_algorithm.combine_type_enum import CombineTypeEnum
from shift_reduce_algorithm.data import Data
from shift_reduce_algorithm.split_type_enum import SplitTypeEnum
from utils.file_util import create_dir


def produce_structure(new_root="", split_direction=SplitTypeEnum.right, global_reverse=True,
                      combine_type=CombineTypeEnum.head_tail):
    import os
    root = "../data/"
    new_root = new_root
    data_types = ["train/", "valid/", "test/"]
    for data_type in data_types:
        samples = []
        for i in range(2, 14):
            filepath = root + data_type + str(i) + "/"
            lists = os.listdir(filepath)  # 列出文件夹下所有的目录与文件
            for i in range(0, len(lists)):
                path = os.path.join(filepath, lists[i])
                if os.path.isfile(path):
                    file_parser = Data(path, split_direction=split_direction, global_reverse=global_reverse,
                                       combine_type=combine_type)
                    file_parser.shift_reduce_golden()
                    samples.extend(file_parser.samples)
        print(samples)
        create_dir(new_root)
        create_dir(new_root + data_type)
        with open(new_root + data_type + "/data.txt", mode='w', encoding='utf-8') as fw:
            lines = ""
            for line in samples:
                lines += line
                lines += "\n"
            lines = lines.strip()
            print(lines)
            fw.write(lines)


if __name__ == '__main__':
    # 标准生产样例
    root = "../dataset/"
    for new_root, reverse, combine_type in [("forward_head_tail/", False, CombineTypeEnum.head_tail)]:
        produce_structure(new_root=root + new_root, global_reverse=reverse, combine_type=combine_type)
