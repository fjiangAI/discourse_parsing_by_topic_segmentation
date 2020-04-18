import keras
import sys

sys.path.append("..")
from model.tm_bert import build_TM_bert_model
from parsing.parsing_index import ParsingIndex
from train_model import nuclearity_tm_bert, relation_tm_bert
from train_model.model_prepare import get_config_path_and_checkpoint_path_and_tokenizer
from train_model.nuclearity_tm_bert import nuclearity_main
from train_model.relation_tm_bert import relation_main
from train_model.structure_tm_bert import structure_main
from utils.file_util import create_dir


def run_nuclearity(root, data_file_path, do_train=True):
    create_dir(root)
    model_root = root + "/tm_bert_nuclearity"
    config_path, checkpoint_path, tokenizer = get_config_path_and_checkpoint_path_and_tokenizer()
    nuclearity_dict = ParsingIndex.nuclearity_dict
    keras.backend.clear_session()
    TM_bert_model = build_TM_bert_model(config_path, checkpoint_path, num_classes=len(nuclearity_dict.items()))
    if do_train:
        nuclearity_main(data_file_path, model_root, TM_bert_model, tokenizer, nuclearity_dict)
    for i in range(0, 5):
        nuclearity_tm_bert.test_model(data_file_path, model_root, TM_bert_model, tokenizer, nuclearity_dict, index=i)


def run_relation(root, data_file_path, do_train=True):
    create_dir(root)
    model_root = root + "/tm_bert_relation"
    relation_dict = ParsingIndex.relation_dict
    config_path, checkpoint_path, tokenizer = get_config_path_and_checkpoint_path_and_tokenizer()
    keras.backend.clear_session()
    TM_bert_model = build_TM_bert_model(config_path, checkpoint_path, num_classes=len(relation_dict.items()))
    if do_train:
        relation_main(data_file_path, model_root, TM_bert_model, tokenizer, relation_dict)
    for i in range(0, 5):
        relation_tm_bert.test_model(data_file_path, model_root, TM_bert_model, tokenizer, relation_dict, index=i)


def run_structure(root, data_file_path, do_train=True):
    create_dir(root)
    model_root = root + "/tm_bert_structure"
    config_path, checkpoint_path, tokenizer = get_config_path_and_checkpoint_path_and_tokenizer()
    structure_dict = ParsingIndex.structure_dict
    keras.backend.clear_session()
    TM_bert_model = build_TM_bert_model(config_path, checkpoint_path, num_classes=len(structure_dict.items()))
    if do_train:
        structure_main(data_file_path, model_root, TM_bert_model, tokenizer, structure_dict)


if __name__ == '__main__':
    root = "./forward_left"
    data_file_path = "../dataset/forward_left"
    #run_structure(root, data_file_path, True)
    run_nuclearity(root, data_file_path, True)
    run_relation(root, data_file_path, True)
