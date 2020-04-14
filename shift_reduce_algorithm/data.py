from bs4 import BeautifulSoup

from shift_reduce_algorithm.model_type_enum import ModelTypeEnum
from shift_reduce_algorithm.node import Node, create_none_node
from shift_reduce_algorithm.shift_reduce import shift, reduce, predict_model_golden, \
    predict_by_model, predict_by_model_n_and_r
from data_structure.selfqueue import SelfQueue
from data_structure.selfstack import SelfStack
from parsing.post_edit import global_optimization
from parsing.parsing_index import ParsingIndex
from shift_reduce_algorithm.split_type_enum import SplitTypeEnum


class Data:
    def __init__(self, file_name, split_direction, combine_type, global_reverse=False):
        self.combine_type = combine_type
        self.file_name = file_name
        self.structure_index = ParsingIndex.structure_index
        self.nucelarity_index = ParsingIndex.nuclearity_index
        self.relation_index = ParsingIndex.relation_index
        self.content = self._get_file_content(file_name)
        self.title = self._get_title()
        self.title_pos = ""
        self.relation_list = self._get_relation_list(split_direction)
        self.edu_queue = self._get_edu_queue(global_reverse)
        self.golden_span = []
        self.model_span = []
        self.samples = []
        self.split_seq = []
        self.sub_queues = []
        self.model_dict = {}
        self.global_reverse = global_reverse
        self.maxlen = 512
        self.tokenizer = None

    @staticmethod
    def _get_file_content(file_name):
        """
        获取文件内容
        :param file_name: 文件名
        :return: None
        """
        content = ""
        with open(file_name, encoding='utf-8', mode='r') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip()
                content = content + line
        return content

    def _get_title(self):
        """
        获取文章的标题
        :return:
        :return:
        """
        soup = BeautifulSoup(self.content, 'xml')
        titles = soup.find_all(name='DISCOURSE')
        title = titles[0]["DiscourseTopic"]
        return title

    def _get_edu_queue(self, global_reverse=False):
        """
        获取EDU的队列使用B4Soup解析
        :param global_reverse: 是否需要翻转EDU的全部顺序
        :return: the queue of edu
        """
        queue = SelfQueue()
        soup = BeautifulSoup(self.content, 'xml')
        p_list = soup.find_all(name='P')
        for p in p_list:
            ID = p["ID"]
            text = p.get_text()
            node = Node()
            node.words = text
            node.start = int(ID)
            node.fromstart = int(ID) - 1
            node.end = int(ID)
            node.fromend = len(p_list) - int(ID)
            # node.segwords()
            queue.add(node)
        if global_reverse:
            queue.reverse()
        return queue

    def _get_relation_list(self, split_direction=SplitTypeEnum.right):
        """
        获取关系列表
        :param split_direction: 切分方式，选择左切分还是右切分
        :return:
        """
        relation_list = []
        # 获取到relation列表
        soup = BeautifulSoup(self.content, 'xml')
        # 获取每个r的段落位置
        r_list = soup.find_all(name='R')  # 找到所有标签名为a的标签
        for r in r_list:
            nuclearity = self.nucelarity_index[int(r['Center']) - 1]
            relation = r['RelationType']
            position = r['ParagraphPosition']
            atoms = position.split('|')
            if len(atoms) < 3:
                start = str(atoms[0]).split('.')[0]
                end = str(atoms[1]).split('.')[-1]
                relation_list.append((start, end, nuclearity, relation))
            else:
                if split_direction == SplitTypeEnum.left:
                    start = str(atoms[0]).split('.')[0]
                    for i in range(1, len(atoms)):
                        end = str(atoms[i]).split('.')[-1]
                        relation_list.append((start, end, nuclearity, relation))
                else:
                    end = str(atoms[-1]).split('.')[-1]
                    for i in range(0, len(atoms) - 1):
                        start = str(atoms[i]).split('.')[0]
                        relation_list.append((start, end, nuclearity, relation))
        relation_list = sorted(relation_list, key=lambda x: (x[0], x[1]))
        return relation_list

    def get_sample(self, node1, node2, node3, lastaction):
        """
            根据当前节点获取样例，这个需要根据模型的输入进行调整。
            :param node1: 第一个节点
            :param node2: 第二个节点
            :param node3: 第三个节点
            :param lastaction: 上一个操作行为
            :return: [X1_token, X1_seg, X2_token, X2_seg, X3_token, X3_seg]
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
        return [X1_token, X1_seg, X2_token, X2_seg, X3_token, X3_seg]

    def _get_samples(self, node1=Node(), node2=Node(), node3=Node(), last_action="s", action="r", nuclearity="None",
                     relation="None"):
        sample = self.file_name + "\t" + self.title + "\t" + \
                 node1.to_string() + "\t" + node2.to_string() + "\t" + node3.to_string() + "\t" + last_action + "\t" + \
                 action + "\t" + nuclearity + "\t" + relation
        self.samples.append(sample)

    def reverse_span(self, span_type="golden"):
        """
        翻转span的开始结束位置
        :param span_type: str 'golden' or 'model'
        :return: None
        """
        new_span = []
        if span_type == "golden":
            for span in self.golden_span:
                new_span.append((span[1], span[0], span[2], span[3], span[4]))
            self.golden_span = new_span
        else:
            for span in self.model_span:
                new_span.append((span[1], span[0], span[2], span[3], span[4]))
            self.model_span = new_span

    def write_span_to_file(self, des_file, span_type="golden"):
        """
        将span结果写入到文件中
        :param des_file: 目标文件
        :param span_type: span的类型 "golden" 或者 "model"
        :return: None
        """
        line = self.file_name + "\t"
        spans = []
        if span_type == "golden":
            spans = self.golden_span
        else:
            spans = self.model_span
        for span in spans:
            if len(span) == 4:
                line = line + "(" + str(span[0]) + "," + str(span[1]) + "," + str(span[2]) + "," + str(span[3]) + ")\t"
            elif len(span) == 5:
                line = line + "(" + str(span[0]) + "," + str(span[1]) + "," + str(span[2]) + "," + str(
                    span[3]) + "," + str(span[4]) + ")\t"
        line = line.strip(",")
        line += "\n"
        with open(des_file, mode='a', encoding='utf-8') as fw:
            fw.write(line)

    def load_model(self, model, model_type):
        self.model_dict[model_type.name] = model

    def load_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def shift_reduce_golden(self):
        """
        创建标准的篇章结构
        :return: None
        """
        stack = SelfStack()
        last_action = "s"
        # 如果队列中还有元素的话
        while self.edu_queue.size() > 0:
            if stack.size() < 2:
                shift(stack, self.edu_queue)
            else:
                action, nuclearity, relation, prob = predict_model_golden(stack.top(2), stack.top(1),
                                                                          self.relation_list,
                                                                          global_reverse=self.global_reverse)
                self._get_samples(stack.top(2), stack.top(1), self.edu_queue.get(), last_action, action, nuclearity,
                                  relation)
                if action == 's':
                    shift(stack, self.edu_queue)
                else:
                    self.golden_span.append(
                        reduce(stack, self.global_reverse, nuclearity, relation, prob, combine_type=self.combine_type))
                last_action = action
        # 如果栈中还有元素的话
        while stack.size() > 1:
            last_action = "s"
            action, nuclearity, relation, prob = predict_model_golden(stack.top(2), stack.top(1), self.relation_list,
                                                                      global_reverse=self.global_reverse)
            self._get_samples(stack.top(2), stack.top(1), create_none_node(), last_action, action, nuclearity, relation)
            self.golden_span.append(
                reduce(stack, self.global_reverse, nuclearity, relation, prob=prob, combine_type=self.combine_type))

    def shift_reduce_model(self):
        """
        使用模型进行shift_reduce实验
        :return:
        """
        stack = SelfStack()
        last_action = "s"
        structure_model = self.model_dict.get(ModelTypeEnum.structure_model.name, None)
        nuclearity_model = self.model_dict.get(ModelTypeEnum.nuclearity_model.name, None)
        relation_model = self.model_dict.get(ModelTypeEnum.relation_model.name, None)
        # 如果队列中还有元素的话
        while self.edu_queue.size() > 0:
            if stack.size() < 2:
                shift(stack, self.edu_queue)
            else:
                # 1.创建测试样例
                sample = self.get_sample(stack.top(2), stack.top(1), self.edu_queue.get(),
                                         last_action)
                # 2.首先预测结构
                action, prob = predict_by_model(sample, structure_model, self.structure_index)
                if action == 's':  # 如果是shift，直接移动
                    shift(stack, self.edu_queue)
                else:  # 3.如果是reduce,则再判断主次和关系
                    nuclearity = predict_by_model_n_and_r(sample, nuclearity_model)
                    relation = predict_by_model_n_and_r(sample, relation_model)
                    # 进行主次优化
                    nuclearity, relation = global_optimization(nuclearity, relation, self.nucelarity_index,
                                                               self.relation_index)
                    self.model_span.append(
                        reduce(stack, self.global_reverse, nuclearity, relation, prob, combine_type=self.combine_type))
                last_action = action
        # 如果栈中还有元素的话
        while stack.size() > 1:
            last_action = "s"
            sample = self.get_sample(stack.top(2), stack.top(1), create_none_node(), last_action)
            action, prob = 'r', 1
            nuclearity = predict_by_model_n_and_r(sample, nuclearity_model)
            relation = predict_by_model_n_and_r(sample, relation_model)
            nuclearity, relation = global_optimization(nuclearity, relation, self.nucelarity_index, self.relation_index)
            self.model_span.append(reduce(stack, self.global_reverse, nuclearity, relation, prob, self.combine_type))

    def load_split_seq(self, seq, global_reverse=False):
        """
        加载分割结果
        :param seq: 分割序列
        :param global_reverse: 是否需要翻转
        :return: None
        """
        self.split_seq = seq
        if global_reverse is True:
            self.split_seq = self.split_seq[::-1]

    def __split_du_by_seq(self):
        """
        根据切分列表切分edu，例如000010001 则会分为5,4两个部分
        :return: None
        """
        assert len(self.split_seq) == len(self.edu_queue.items)
        sub_queue = SelfQueue()
        for i in range(0, len(self.split_seq)):
            sub_queue.add(self.edu_queue.items[i])
            if self.split_seq[i] == 0:
                # 如果这个位置是0，则不动
                pass
            else:
                # 如果这个位置是1，则需要重新一个子序列
                self.sub_queues.append(sub_queue)
                sub_queue = SelfQueue()
        self.sub_queues.append(sub_queue)

    def shift_reduce_model_lower_level(self):
        """
        先进行第一步,在每个子序列中进行parsing
        :return:
        """
        self.__split_du_by_seq()  # 拆分出子序列
        high_dus = SelfQueue()
        for sub_queue in self.sub_queues:
            high_du, sub_model_span = self.__shift_reduce_model_sub_tree(sub_queue)
            high_dus.add(high_du)
            self.model_span.extend(sub_model_span)
        self.edu_queue = high_dus

    def __shift_reduce_model_sub_tree(self, sub_queue=SelfQueue()):
        """
        对于子树进行shift-reduce切分
        :param sub_queue: 待切分的子树
        :return: 最后的节点，span
        """
        # 如果队列中还有元素的话
        stack = SelfStack()
        last_action = "s"
        sub_model_span = []
        structure_model = self.model_dict.get(ModelTypeEnum.structure_model.name, None)
        nuclearity_model = self.model_dict.get(ModelTypeEnum.nuclearity_model.name, None)
        relation_model = self.model_dict.get(ModelTypeEnum.relation_model.name, None)
        if sub_queue.size() == 1:
            return sub_queue.put(), sub_model_span
        else:
            while sub_queue.size() > 0:
                if stack.size() < 2:
                    shift(stack, sub_queue)
                else:
                    # 1.创建测试样例
                    sample = self.get_sample(stack.top(2), stack.top(1), self.edu_queue.get(), last_action)
                    # 2.首先预测结构
                    action, prob = predict_by_model(sample, structure_model, self.structure_index)
                    if action == 's':  # 如果是shift，直接移动
                        shift(stack, sub_queue)
                    else:  # 3.如果是reduce,则再判断主次和关系
                        nuclearity = predict_by_model_n_and_r(sample, nuclearity_model)
                        relation = predict_by_model_n_and_r(sample, relation_model)
                        nuclearity, relation = global_optimization(nuclearity, relation, self.nucelarity_index,
                                                                   self.relation_index)
                        sub_model_span.append(reduce(stack, self.global_reverse, nuclearity, relation, prob))
                    last_action = action
            # 如果栈中还有元素的话
            while stack.size() > 1:
                last_action = "s"
                sample = self.get_sample(stack.top(2), stack.top(1), create_none_node(), last_action)
                action, prob = 'r', 1
                nuclearity = predict_by_model_n_and_r(sample, nuclearity_model)
                relation = predict_by_model_n_and_r(sample, relation_model)
                nuclearity, relation = global_optimization(nuclearity, relation, self.nucelarity_index,
                                                           self.relation_index)
                sub_model_span.append(reduce(stack, self.global_reverse, nuclearity, relation, prob))
            return stack.pop(), sub_model_span
