import numpy as np


def get_max_row_and_col_index(result_matrix):
    re = np.where(result_matrix == np.max(result_matrix))
    if len(re[0]) == 1:
        return re[0][0], re[1][0]
    else:
        return re[0][0], re[0][1]


def global_optimization(nuclearity_matrix, relation_matrix, nuclearity_index={}, relation_index={}):
    """
    use mask to optimize nuclearity and relation.
    :param nuclearity_matrix:
    :param relation_matrix:
    :param nuclearity_index:
    :param relation_index:
    :return: best nuclearity and relation
    """
    mask_matrix = [[0, 0, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 0, 0],
                   [1, 0, 1],
                   [1, 1, 1],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 1, 0],
                   [1, 1, 0],
                   [1, 1, 0]]
    relation_nuclearity_matrix = np.dot(relation_matrix.T, nuclearity_matrix)
    result_matrix = np.multiply(relation_nuclearity_matrix, mask_matrix)
    max_relation_index, max_nuclearity_index = get_max_row_and_col_index(result_matrix)
    relation_action = relation_index[max_relation_index]
    nuclearity_action = nuclearity_index[max_nuclearity_index]
    return nuclearity_action, relation_action
