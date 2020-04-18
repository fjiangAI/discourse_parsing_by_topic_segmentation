import keras
import sys
sys.path.append("..")
from baselines.bert_baseline import structure_main
from baselines import bert_baseline_n
from baselines import bert_baseline_r
from baselines.bert_baseline_n import nuclearity_main
from baselines.bert_baseline_r import relation_main
from model.bert import build_bert_model
from parsing.parsing_index import ParsingIndex
from train_model.model_prepare import get_config_path_and_checkpoint_path_and_tokenizer
from utils.file_util import create_dir


def run_nuclearity(root, data_file_path):
    root = root
    create_dir(root)
    data_file_path = data_file_path
    model_root = root + "/bert_nuclearity"
    keras.backend.clear_session()
    config_path, checkpoint_path, tokenizer = get_config_path_and_checkpoint_path_and_tokenizer()
    nuclearity_dict = ParsingIndex.nuclearity_dict
    bert_model = build_bert_model(config_path, checkpoint_path, num_classes=len(nuclearity_dict.items()))
    nuclearity_main(data_file_path, model_root, bert_model, tokenizer, nuclearity_dict)
    for i in range(0, 5):
        bert_baseline_n.test_model(data_file_path, model_root, bert_model, tokenizer, nuclearity_dict, index=i)


def run_relation(root, data_file_path):
    root = root
    create_dir(root)
    data_file_path = data_file_path
    model_root = root + "/bert_relation"
    keras.backend.clear_session()
    config_path, checkpoint_path, tokenizer = get_config_path_and_checkpoint_path_and_tokenizer()
    relation_dict = ParsingIndex.relation_dict
    bert_model = build_bert_model(config_path, checkpoint_path, num_classes=len(relation_dict.items()))
    relation_main(data_file_path, model_root, bert_model, tokenizer, relation_dict)
    for i in range(0, 5):
        bert_baseline_r.test_model(data_file_path, model_root, bert_model, tokenizer, relation_dict, index=i)


def run_structure(root, data_file_path):
    root = root
    create_dir(root)
    data_file_path = data_file_path
    config_path, checkpoint_path, tokenizer = get_config_path_and_checkpoint_path_and_tokenizer()
    structure_dict = ParsingIndex.structure_dict
    for save_model_root in [
        root + "/bert_structure"
    ]:
        keras.backend.clear_session()
        bert_model = build_bert_model(config_path, checkpoint_path, num_classes=len(structure_dict.items()))
        structure_main(data_file_path, save_model_root, bert_model, tokenizer, structure_dict)


if __name__ == '__main__':
    root = "./ forward_left"
    data_file_path = "../dataset/forward_left"
    run_structure(root, data_file_path)
    run_nuclearity(root, data_file_path)
    run_relation(root, data_file_path)
