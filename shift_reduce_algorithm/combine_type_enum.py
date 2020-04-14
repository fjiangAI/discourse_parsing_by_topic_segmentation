from enum import Enum


class CombineTypeEnum(Enum):
    """
    合并篇章单元的方式
    """
    left = 0
    right = 1
    all = 2
    all_reverse = 3
    head_tail = 4
    head_tail_reverse = 5
    two_head = 6
    two_tail = 7
    two_head_reverse = 8
    two_tail_reverse = 9
