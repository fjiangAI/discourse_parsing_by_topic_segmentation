from enum import Enum


class ModelTypeEnum(Enum):
    """
    parsing需要加载的模型
    """
    structure_model = 0,
    nuclearity_model = 1,
    relation_model = 2
