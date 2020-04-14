import jieba.posseg as psg

from data_structure.selfstack import SelfStack
from shift_reduce_algorithm.combine_type_enum import CombineTypeEnum


class Node:
    def __init__(self):
        self.words = ""
        self.poss = ""
        self.start = 0
        self.end = 0
        self.from_start = 0
        self.from_end = 0
        self.size = 1
        self.start_words = ""
        self.start_poss = ""
        self.end_words = ""
        self.end_poss = ""
        self.children = []

    def seg_words(self):
        """
        分词
        :return: None
        """
        new_words = ""
        new_poss = ""
        for x in psg.cut(self.words):
            new_words = new_words + x.word + " "
            new_poss = new_poss + x.flag + " "
        new_words = new_words.strip()
        new_poss = new_poss.strip()
        self.words = new_words
        self.poss = new_poss

    def set_other_features(self):
        """
        设置其他的特征
        :return: None
        """
        self.start = self.children[0].start
        self.end = self.children[1].end
        self.from_start = self.children[0].from_start
        self.from_end = self.children[1].from_end
        self.size = self.children[0].size + self.children[1].size

    def get_left_most_node(self):
        """
        获取最左边的节点（头节点）
        :return:
        """
        cur_node = self.children[0]
        while len(cur_node.children) != 0:
            cur_node = cur_node.children[0]
        return cur_node

    def get_right_most_node(self):
        """
        获取最右边的节点（尾节点）
        :return:
        """
        cur_node = self.children[1]
        while len(cur_node.children) != 0:
            cur_node = cur_node.children[1]
        return cur_node

    def get_head_and_tail_node(self, reverse=False):
        """
        获取头节点和尾节点
        :param reverse: 是否翻转
        :return:
        """
        head_node = self.get_left_most_node()
        tail_node = self.get_right_most_node()
        if reverse:
            return tail_node, head_node
        else:
            return head_node, tail_node

    def get_all_nodes(self, reverse=False):
        """
        使用深度遍历获取所有节点
        :return: list
        """
        node_stack = SelfStack()
        all_nodes_list = []
        node_stack.push(self.children[1])
        node_stack.push(self.children[0])
        while not node_stack.empty():
            cur_node = node_stack.pop()
            if len(cur_node.children) == 0:  # 是叶子节点，加入候选序列
                all_nodes_list.append(cur_node)
            else:
                node_stack.push(cur_node.children[1])
                node_stack.push(cur_node.children[0])
        if reverse:
            all_nodes_list = all_nodes_list.reverse()
        return all_nodes_list

    def to_string(self):
        return str(self.words) + "\t" + \
               str(self.start) + "\t" + str(self.end) + "\t" + \
               str(self.from_start) + "\t" + str(self.from_end) + "\t" + str(self.size)

    def set_words_and_poss(self, candidate_node_list=[]):
        """
        进行文字提取
        :param candidate_node_list:
        :return:
        """
        for candidate_node in candidate_node_list:
            self.words = self.words + candidate_node.words
            self.poss = self.poss + candidate_node.poss


def combination(node1=Node(), node2=Node(), nuclearity="NS", global_reverse=False, combine_type="left"):
    """
    拼接两个篇章单元,如果左边重要，则文字选择左边的作为代表，
    如果右边重要，则选择右边作为代表，
    如果同等重要，则选取左边的。
    :param combine_type:
    :param global_reverse:
    :param nuclearity:
    :param node1:篇章单元1
    :param node2:篇章单元2
    :return: 新的篇章单元
    """
    node = Node()
    node.children.append(node1)
    node.children.append(node2)
    node.set_other_features()
    candidate_node_list = []
    if global_reverse is False:
        if combine_type == CombineTypeEnum.left:
            candidate_node_list.append(node.get_left_most_node())
        elif combine_type == CombineTypeEnum.right:
            candidate_node_list.append(node.get_left_most_node())
        elif combine_type == CombineTypeEnum.all:
            candidate_node_list = node.get_all_nodes()
        elif combine_type == CombineTypeEnum.all_reverse:
            candidate_node_list = node.get_all_nodes(reverse=True)
        elif combine_type == CombineTypeEnum.head_tail:
            head_node, tail_node = node.get_head_and_tail_node()
            candidate_node_list.append(head_node)
            candidate_node_list.append(tail_node)
        elif combine_type == CombineTypeEnum.head_tail_reverse:
            tail_node, head_node = node.get_head_and_tail_node(reverse=True)
            candidate_node_list.append(tail_node)
            candidate_node_list.append(head_node)
    else:
        # todo
        # 全局翻转的还没有做
        pass
    node.set_words_and_poss(candidate_node_list)
    return node


def create_none_node():
    node = Node()
    node.words = "None"
    node.poss = ""
    node.start = -1
    node.end = -1
    node.from_start = -1
    node.from_end = -1
    node.size = -1
    return node
