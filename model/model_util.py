import codecs

from keras_bert import Tokenizer
import numpy as np

class OurTokenizer(Tokenizer):
    """
    自定义分词表
    """

    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def get_tokenizer(dict_path):
    """
    获取分词器
    :param dict_path:
    :return:
    """
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = OurTokenizer(token_dict)
    return tokenizer


def seq_padding(X, padding=0):
    """
    实现补0策略
    :param X:
    :param padding:
    :return:
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])