from model.model_util import get_tokenizer
from pretrain_model.bert_config import BertConfig


def get_config_path_and_checkpoint_path_and_tokenizer():
    config_path = BertConfig.config_path
    checkpoint_path = BertConfig.checkpoint_path
    vocab_path = BertConfig.vocab_path
    print("加载分词器")
    tokenizer = get_tokenizer(vocab_path)
    return config_path, checkpoint_path, tokenizer
