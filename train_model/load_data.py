def get_data_by_file(file_name):
    """
    根据文件创建两两样例
    :param file_name:
    :return:
    """
    ds = []
    with open(file_name, mode='r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            d = {}
            line = line.strip()
            items = line.split('\t')
            d["file_name"] = items[0]
            d["title"] = items[1]
            d["arg0"] = items[2]
            d["arg0_position"] = items[3]
            d["arg0_fromstart"] = items[5]
            d["arg0_fromend"] = items[6]
            d["arg1"] = items[8]
            d["arg2"] = items[14]
            d["structure"] = items[21]
            d["nuclearity"] = items[22]
            d["relation"] = items[23]
            ds.append(d)
    return ds


def get_data(root):
    """
    获取训练数据和测试数据
    :param root:
    :return:
    """
    train_root = root + "/train/data.txt"
    valid_root = root + "/valid/data.txt"
    test_root = root + "/test/data.txt"
    train_ds = get_data_by_file(train_root)
    valid_ds = get_data_by_file(valid_root)
    test_ds = get_data_by_file(test_root)
    new_train = []
    new_train.extend(train_ds)
    new_train.extend(valid_ds)
    return new_train, test_ds


def del_none_data(data, datatype="nuclearity"):
    new_data = []
    for d in data:
        if d[datatype] == "None":
            continue
        else:
            new_data.append(d)
    return new_data


def get_test_golden_label(test_data, relation_dict, datatype="nuclearity"):
    labels = []
    print(relation_dict)
    for d in test_data:
        if d[datatype] == "None":
            continue
        else:
            y = relation_dict[d[datatype]]
            labels.append(y)
    return labels
