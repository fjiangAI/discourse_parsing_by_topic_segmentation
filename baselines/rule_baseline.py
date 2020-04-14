from utils.file_util import get_file_list


def create_rule_span(start, end):
    strings = "(" + str(start) + "," + str(end) + ",NS,Elaboration,1)"
    return strings


def create_rule_result(file, branch_type="left"):
    length = int(file.split("/")[3])
    result = file
    if branch_type == "left":
        for start in [1]:
            for end in range(2, length + 1):
                result = result + "\t" + create_rule_span(start, end)
    else:
        for start in range(1, length):
            for end in [length]:
                result = result + "\t" + create_rule_span(start, end)
    return result


def get_result_by_path_rule(path, test_result, branch_type="left"):
    files = get_file_list(path)
    with open(test_result, encoding='utf-8', mode="a") as fw:
        for file in files:
            print("开始测试:" + file)
            fw.write(create_rule_result(file, branch_type) + "\n")


def get_result_by_root_rule(root, test_result, branch_type="left"):
    for i in range(2, 14):
        path = root + str(i) + "/"
        get_result_by_path_rule(path, test_result, branch_type)


def main_rule(branch_type="left"):
    root = "./data/test/"
    test_result = "./规则/test_" + str(branch_type) + ".txt"
    print("开始测试")
    get_result_by_root_rule(root, test_result, branch_type=branch_type)
