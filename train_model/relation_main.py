from keras.optimizers import Adam

from model.tm_bert import build_TM_bert_model
from train_model.load_data import get_data, del_none_data, get_test_golden_label

from model.tm_bert import DataGenerator
from parsing.parsing_index import ParsingIndex
from train_model.model_prepare import get_config_path_and_checkpoint_path_and_tokenizer
from utils.file_util import create_dir
from utils.metric_util import write_result


def relation_main(data_file_name, save_model_root, model, tokenizer, relation_dict):
    maxlen = 512
    epoch = 5
    train_data, test_data = get_data(data_file_name)
    print("完成数据加载")
    train_data = del_none_data(train_data, datatype="relation")  # 这句与结构不同
    train_d = DataGenerator(train_data, batch_size=2,
                            relation_dict=relation_dict,
                            tokenizer=tokenizer,
                            maxlen=maxlen,
                            data_type="relation")
    print("完成迭代器转换")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    create_dir(save_model_root)
    print("完成模型构建")
    for i in range(epoch):
        model.fit_generator(
            train_d.__iter__(),
            steps_per_epoch=len(train_d)
        )
        save_file = save_model_root + "/save_model" + str(i) + "epoch.model"
        model.save_weights(save_file)


def test_model(data_file_name, load_model_root, model, tokenizer, relation_dict, index=1):
    maxlen = 512
    train_data, test_data = get_data(data_file_name)
    test_data = del_none_data(test_data, datatype="relation")
    test_d = DataGenerator(test_data, batch_size=2, relation_dict=relation_dict, tokenizer=tokenizer, maxlen=maxlen,
                           data_type="relation")
    model_name = load_model_root + "/save_model" + str(index) + "epoch.model"
    model.load_weights(model_name)
    temp_result = model.predict_generator(test_d.__iter__(), steps=len(test_d)).argmax(axis=-1)
    golden_label = get_test_golden_label(test_data, relation_dict, datatype="relation")
    output_golden_file = load_model_root + "/result" + str(index) + ".txt"
    relation_list = ['Joint',
                     'Sequence',
                     'Progression',
                     "Contrast",
                     "Supplement",
                     "Cause-Result",
                     "Result-Cause",
                     "Background",
                     "Behavior-Purpose",
                     "Purpose-Behavior",
                     "Elaboration",
                     "Summary",
                     "Evaluation",
                     "Statement-Illustration",
                     "Illustration-Statement"
                     ]
    write_result(output_golden_file, golden_label, temp_result, label_name=relation_list)
    del model


if __name__ == '__main__':
    import keras

    root = "./output"
    create_dir(root)
    data_file_name = ""
    model_root = ""
    relation_dict = ParsingIndex.relation_dict
    config_path, checkpoint_path, tokenizer = get_config_path_and_checkpoint_path_and_tokenizer()
    keras.backend.clear_session()
    TM_bert_model = model = build_TM_bert_model(config_path, checkpoint_path, num_classes=len(relation_dict.items()))
    relation_main(data_file_name, model_root, TM_bert_model, tokenizer, relation_dict)
    for i in range(0, 5):
        test_model(data_file_name, model_root, TM_bert_model, tokenizer, relation_dict, index=i)
