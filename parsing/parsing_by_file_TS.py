import sys

sys.path.append("..")
from parsing.parsing_index import ParsingIndex
from pretrain_model.bert_config import BertConfig
from shift_reduce_algorithm.combine_type_enum import CombineTypeEnum
from model.tm_bert import build_TM_bert_model
from shift_reduce_algorithm.data import Data
from model.model_util import get_tokenizer

from shift_reduce_algorithm.model_type_enum import ModelTypeEnum
from shift_reduce_algorithm.split_type_enum import SplitTypeEnum
from utils.file_util import get_file_list, create_dir


def load_seq_file(file_name):
    """
    加载序列文件
    :param file_name: 保存序列的文件
    :return:
    """
    seq_dict = {}
    with open(file_name, encoding='utf-8', mode='r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            items = line.split('\t')
            file_name = items[0]
            seq = "".join(items[1:])
            seq_dict[file_name] = seq
    return seq_dict


def get_split_seq(goal_file_name, seq_dict={}):
    """
    获得目标序列
    :param goal_file_name:
    :param seq_dict:
    :return:
    """
    for name, seq in seq_dict.items():
        if name in goal_file_name:
            return seq
    return None


def get_result_by_path_topic_segmentation(path, golden_result, test_result,
                                          tokenizer,
                                          structure_model, nuclearity_model, relation_model,
                                          combine_type,
                                          global_reverse):
    files = get_file_list(path)
    seq_file_name = "./all_segmentation_golden_boundary.txt"
    seq_dict = load_seq_file(seq_file_name)
    for file in files:
        print("开始测试:" + file)
        # 生成黄金结果
        file_golden = Data(file, split_direction=SplitTypeEnum.right, global_reverse=global_reverse,
                           combine_type=combine_type)
        file_golden.shift_reduce_golden()
        file_golden.write_span_to_file(golden_result)
        # 生成测试结果
        file_test = Data(file, split_direction=SplitTypeEnum.right, global_reverse=global_reverse,
                         combine_type=combine_type)
        file_test.load_tokenizer(tokenizer=tokenizer)
        file_test.load_model(structure_model, model_type=ModelTypeEnum.structure_model)
        file_test.load_model(nuclearity_model, model_type=ModelTypeEnum.nuclearity_model)
        file_test.load_model(relation_model, model_type=ModelTypeEnum.relation_model)
        seq = get_split_seq(file_test.file_name, seq_dict)
        file_test.load_split_seq(seq=seq)
        file_test.shift_reduce_model_lower_level()
        file_test.shift_reduce_model()
        file_test.write_span_to_file(test_result, span_type="test")


def get_result_by_root_topic_segmentation(dataset_root, golden_result, test_result,
                                          tokenizer, structure_model, nuclearity_model, relation_model,
                                          combine_type,
                                          global_reverse=False):
    for i in range(2, 14):
        path = dataset_root + str(i) + "/"
        get_result_by_path_topic_segmentation(path, golden_result, test_result,
                                              tokenizer, structure_model, nuclearity_model, relation_model,
                                              combine_type=combine_type, global_reverse=global_reverse)


def load_data(model_file, config_path, checkpoint_path, num_classes):
    model = build_TM_bert_model(config_path, checkpoint_path, num_classes=num_classes)
    model.load_weights(model_file)
    return model


def create_golden_and_test_result_file_name(model_root, model_name):
    golden_result = model_root + "/golden.txt"
    test_result = model_root + "/test_" + str(model_name) + ".txt"
    return golden_result, test_result


def main_combine_topic_segmentation(model_name, combine_type,
                                    structure_model_file,
                                    nuclearity_model_file,
                                    relation_model_file,
                                    global_reverse=False,
                                    dataset_root="../data/test/"):
    model_root = "./" + model_name
    create_dir(model_root)
    golden_result_file_name, test_result_file_name = create_golden_and_test_result_file_name(model_root, model_name)
    config_path = BertConfig.config_path
    checkpoint_path = BertConfig.checkpoint_path
    vocab_path = BertConfig.vocab_path
    print("加载分词器")
    tokenizer = get_tokenizer(vocab_path)
    print("加载结构模型")
    structure_model = load_data(structure_model_file, config_path, checkpoint_path,
                                num_classes=len(ParsingIndex.structure_index.keys()))
    print("加载主次模型")
    nuclearity_model = load_data(nuclearity_model_file, config_path, checkpoint_path,
                                 num_classes=len(ParsingIndex.nuclearity_index.keys()))
    print("加载关系模型")
    relation_model = load_data(relation_model_file, config_path, checkpoint_path,
                               num_classes=len(ParsingIndex.relation_index.keys()))
    print("开始测试")
    get_result_by_root_topic_segmentation(dataset_root, golden_result_file_name, test_result_file_name,
                                          tokenizer, structure_model, nuclearity_model, relation_model,
                                          combine_type=combine_type, global_reverse=global_reverse)


if __name__ == '__main__':
    import keras

    structure_model_file = "../train_model/forward_left/structure/save_model4epoch.model"
    nuclearity_model_file = "../train_model/forward_left/nuclearity/save_model4epoch.model"
    relation_model_file = "../train_model/forward_left/relation/save_model4epoch.model"
    for model_name, combine_type, reverse in [
        ("forward_left_TS", CombineTypeEnum.left, False)
    ]:
        keras.backend.clear_session()
        main_combine_topic_segmentation(model_name=model_name,
                                        combine_type=combine_type,
                                        structure_model_file=structure_model_file,
                                        nuclearity_model_file=nuclearity_model_file,
                                        relation_model_file=relation_model_file,
                                        global_reverse=reverse)
