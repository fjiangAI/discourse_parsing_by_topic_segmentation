from keras.optimizers import Adam
from train_model.load_data import get_data

from model.tm_bert import DataGenerator, build_TM_bert_model
from parsing.parsing_index import ParsingIndex
from train_model.model_prepare import get_config_path_and_checkpoint_path_and_tokenizer
from utils.file_util import create_dir


def structure_main(data_file_name, save_model_root, model, tokenizer, structure_dict):
    """
    结构模型训练主程序
    :param structure_dict: 标签字典
    :param data_file_name: 训练、测试数据路径
    :param save_model_root: 保存模型路径
    :param model: 要加载的模型
    :param tokenizer: token
    :return: None
    """
    maxlen = 512
    epoch = 5
    print(data_file_name)
    train_data, test_data = get_data(data_file_name)
    print("完成数据加载")
    train_d = DataGenerator(train_data, batch_size=2,
                            relation_dict=structure_dict,
                            tokenizer=tokenizer,
                            maxlen=maxlen)
    print("完成迭代器转换")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    print("完成模型构建")
    create_dir(save_model_root)
    for i in range(epoch):
        model.fit_generator(
            train_d.__iter__(),
            steps_per_epoch=len(train_d)
        )
        save_file = save_model_root + "/save_model" + str(i) + "epoch.model"
        model.save_weights(save_file)
    del model


if __name__ == '__main__':
    import keras

    data_file_name = ""
    root = "./output"
    create_dir(root)
    config_path, checkpoint_path, tokenizer = get_config_path_and_checkpoint_path_and_tokenizer()
    structure_dict = ParsingIndex.structure_dict
    for save_model_root in [
        root + "/forward_head_tail"
    ]:
        keras.backend.clear_session()
        TM_bert_model = build_TM_bert_model(config_path, checkpoint_path, num_classes=len(structure_dict.items()))
        structure_main(data_file_name, save_model_root, TM_bert_model, tokenizer, structure_dict)
