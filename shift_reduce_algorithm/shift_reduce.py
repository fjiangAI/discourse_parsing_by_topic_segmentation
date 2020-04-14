from data_structure.selfqueue import SelfQueue
from data_structure.selfstack import SelfStack
from shift_reduce_algorithm.node import combination, Node


def shift(stack=SelfStack(), queue=SelfQueue()):
    last = queue.put()
    stack.push(last)


def reduce(stack=SelfStack(), reverse=False, nuclearity="None", relation="None", prob=1, combine_type="right"):
    top2 = stack.pop()
    top1 = stack.pop()
    new_top = combination(top1, top2, nuclearity, reverse, combine_type=combine_type)
    stack.push(new_top)
    span = (new_top.start, new_top.end, nuclearity, relation, prob)
    return span


def predict_model_golden(node1=Node(), node2=Node(), relation_list=[], global_reverse=False):
    action = 's'
    nucleartiy = "None"
    relation = "None"
    for r in relation_list:
        if not global_reverse:
            if int(node1.start) == int(r[0]) and int(node2.end) == int(r[1]):
                action = "r"
                nucleartiy = r[2]
                relation = r[3]
                break
        else:
            if int(node2.start) == int(r[0]) and int(node1.end) == int(r[1]):
                action = "r"
                nucleartiy = r[2]
                relation = r[3]
                break
    return action, nucleartiy, relation, 1


def predict_by_model(sample, model=None, relation_index={}):
    """
    使用模型预测
    :param sample:
    :param model:
    :param relation_index:
    :return:
    """
    result_y = model.predict(sample).argmax(axis=-1)
    max_prob = max(model.predict(sample)[0])
    label = result_y[0]
    action = relation_index[label]
    return action, max_prob


def predict_by_model_n_and_r(sample, model=None):
    """
    为了主次和关系的联合预测
    :param sample: 样本
    :param model: 模型
    :return:
    """
    result_y = model.predict(sample)
    return result_y
