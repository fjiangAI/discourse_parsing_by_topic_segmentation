from shift_reduce_algorithm.data import Data


class BertData(Data):
    def __init__(self, file_name, split_direction, combine_type, global_reverse=False):
        super().__init__(file_name, split_direction, combine_type, global_reverse)

    def get_sample(self, node1, node2, node3, lastaction):
        """
            根据当前节点获取样例，这个需要根据模型的输入进行调整。
            :param node1: 第一个节点
            :param node2: 第二个节点
            :param node3: 第三个节点
            :param lastaction: 上一个操作行为
            :return: [X1_token, X1_seg]
            """
        X1_token, X1_seg, \
        X2_token, X2_seg, \
        X3_token, X3_seg, \
        X1_position, X1_fromstart, X1_fromend = [], [], [], [], [], [], [], [], []
        d = {}
        d['arg0'] = node1.to_string().split("\t")[0]
        d['arg1'] = node2.to_string().split("\t")[0]
        d['arg2'] = node3.to_string().split("\t")[0]
        arg1_text = d['arg0'][:self.maxlen]
        arg2_text = d['arg1'][:self.maxlen]
        arg3_text = d['arg2'][:self.maxlen]
        x1_token, x1_seg = self.tokenizer.encode(first=arg1_text + arg2_text + arg3_text,
                                                 max_len=self.maxlen)
        X1_token.append(x1_token)
        X1_seg.append(x1_seg)
        return [X1_token, X1_seg]
