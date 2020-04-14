import os
import sys
from keras import Input
from keras.layers import Lambda, Dense, concatenate, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_bert import load_trained_model_from_checkpoint
from keras.models import Model
from model.model_util import seq_padding
import numpy as np


class DataGenerator:
    """
    生成迭代数据
    """

    def __init__(self, data, batch_size=8, tokenizer=None, maxlen=512, relation_dict={}, data_type="structure"):
        """

        :param data:
        :param batch_size:
        :param tokenizer:
        :param maxlen:
        :param relation_dict:
        """
        self.data = data
        self.datatype = data_type
        self.batch_size = batch_size
        self.num_classes = len(relation_dict.items())
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.relation_dict = relation_dict
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = np.arange(len(self.data))
            np.random.shuffle(idxs)
            X1_token, X1_seg, \
            X2_token, X2_seg, \
            X3_token, X3_seg, Y = [], [], [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                arg1_text = d['arg0'][:self.maxlen]
                arg2_text = d['arg1'][:self.maxlen]
                arg3_text = d['arg2'][:self.maxlen]
                title = d['title'][:self.maxlen]
                x1_token, x1_seg = self.tokenizer.encode(first=arg1_text, second=arg2_text,
                                                         max_len=self.maxlen)
                X1_token.append(x1_token)
                X1_seg.append(x1_seg)
                x2_token, x2_seg = self.tokenizer.encode(first=arg2_text, second=arg3_text,
                                                         max_len=self.maxlen)
                X2_token.append(x2_token)
                X2_seg.append(x2_seg)
                x3_token, x3_seg = self.tokenizer.encode(first=arg1_text, second=arg3_text,
                                                         max_len=self.maxlen)
                X3_token.append(x3_token)
                X3_seg.append(x3_seg)
                y = self.relation_dict[d[self.datatype]]
                Y.append(y)
                if len(X1_token) == self.batch_size or i == idxs[-1]:
                    X1_token = seq_padding(X1_token)
                    X1_seg = seq_padding(X1_seg)
                    X2_token = seq_padding(X2_token)
                    X2_seg = seq_padding(X2_seg)
                    X3_token = seq_padding(X3_token)
                    X3_seg = seq_padding(X3_seg)
                    Y = to_categorical(np.array(Y), num_classes=self.num_classes)
                    yield [X1_token, X1_seg, X2_token, X2_seg, X3_token, X3_seg], Y
                    [X1_token, X1_seg,
                     X2_token, X2_seg,
                     X3_token, X3_seg,
                     Y] = [], [], [], [], [], [], []


def build_TM_bert_model(config_path="", checkpoint_path="", num_classes=2):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True
    x1_token = Input(shape=(None,))
    x1_seg = Input(shape=(None,))
    x2_token = Input(shape=(None,))
    x2_seg = Input(shape=(None,))
    x3_token = Input(shape=(None,))
    x3_seg = Input(shape=(None,))
    x12 = bert_model([x1_token, x1_seg])
    x23 = bert_model([x2_token, x2_seg])
    x13 = bert_model([x3_token, x3_seg])
    x12 = Lambda(lambda x: x[:, 0])(x12)
    x23 = Lambda(lambda x: x[:, 0])(x23)
    x13 = Lambda(lambda x: x[:, 0])(x13)
    x = concatenate([x12, x23, x13])
    p = Dense(num_classes, activation='softmax')(x)

    model = Model([x1_token, x1_seg, x2_token, x2_seg, x3_token, x3_seg], p)
    return model