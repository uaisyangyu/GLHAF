import json
import os
from distutils.command.config import config
from pathlib import Path
from easydict import EasyDict as edict


def get_config_regression(model_name, dataset_name, config_file=""):
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        config_file (str): Path to config file, if given an empty string, will use default config file.
        model_name (str): Name of model.
        dataset_name (str): Name of dataset.

    Returns:
        config (dict): config of the given dataset and model
    """
    #如果 config_file 参数为空字符串，则设置为默认路径。默认路径是当前文件所在目录的 config 文件夹下的 config_regression.json 文件。
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']#json文件中"commonParams"一系列参数
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]#json文件中"datasetParams"的两个数据集其中一个参数
    dataset_args = config_all['datasetCommonParams'][dataset_name]#原来两个数据集的参数
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    config = edict(config)


    return config