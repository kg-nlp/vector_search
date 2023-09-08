# -*- coding:utf-8 -*-
'''
# @FileName    :piplines.py
---------------------------------
# @Time        :2023/9/8 
# @Author      :zhangyj-n
# @Email       :13091375161@163.com
# @zhihu       :https://www.zhihu.com/people/zhangyj-n
# @kg-nlp      :https://kg-nlp.github.io/Algorithm-Project-Manual/
---------------------------------
# 目标任务 :   执行tool.vector_extract,recall,evaluate
---------------------------------
'''

from tool.vector_extract import get_vector
from recall import get_dev_result
from evaluate import evaluate_result
from loguru import logger
import os
import time
current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(current_dir,'data')
result_dir = os.path.join(current_dir,'recall_result_dir')


if __name__ == '__main__':
    # model_name = 'm3e-small'  # 512
    # model_name = 'm3e-base'  # 768
    # model_name = 'm3e-large'  # 1024

    # model_name = 'bge-small-zh'  # 512
    # model_name = 'bge-base-zh'  # 768
    # model_name = 'bge-large-zh'  # 1024

    # model_name = 'text2vec-base-chinese'  # 768
    # model_name = 'text2vec-large-chinese'  # 1024
    # model_name = 'text2vec-base-chinese-sentence'  # 768
    # model_name = 'text2vec-base-chinese-paraphrase'  # 768
    # model_name = 'text2vec-bge-large-chinese'  # 1024

    # model_name = 'Ernie-model/ernie-1.0-base-zh'  # 768
    # model_name = 'Ernie-model/ernie-1.0-large-zh-cw'  # 1024
    # model_name = 'Ernie-model/rocketqa-zh-dureader-query-encoder'  # 768
    # model_name = 'Ernie-model/ernie-3.0-medium-zh'  #  768
    # model_name = 'Ernie-model/ernie-3.0-base-zh'  #  768
    # model_name = 'Ernie-model/rocketqa-zh-base-query-encoder'  # 768

    model_config = {
        # 'm3e-small': 'torch',
        # 'm3e-base': 'torch',
        # 'm3e-large': 'torch',
        # 'bge-small-zh': 'torch',
        # 'bge-base-zh': 'torch',
        # 'bge-large-zh': 'torch',
        # 'text2vec-base-chinese': 'torch',
        # 'text2vec-large-chinese': 'torch',
        # 'text2vec-base-chinese-sentence': 'torch',
        # 'text2vec-base-chinese-paraphrase': 'torch',
        # 'text2vec-bge-large-chinese': 'torch',
        'Ernie-model/ernie-1.0-base-zh':'paddle',
        'Ernie-model/ernie-1.0-large-zh-cw':'paddle',
        "Ernie-model/ernie-3.0-medium-zh": 'paddle',
        'Ernie-model/ernie-3.0-base-zh': 'paddle',
        'Ernie-model/rocketqa-zh-dureader-query-encoder': 'paddle',
        'Ernie-model/rocketqa-zh-base-query-encoder': 'paddle'
    }
    tool_type = ['hnswlib','faiss','milvus','es']

    corpus_file = os.path.join(data_dir, 'corpus.tsv')
    query_file = os.path.join(data_dir, 'dev.tsv')
    recall_result_file = os.path.join(result_dir, 'recall_result.txt')

    for model_name,model_type in model_config.items():
        logger.info('+'*100)
        logger.info('处理模型{}'.format(model_name))
        output_file = os.path.join(data_dir, model_name + '.npy')
        # logger.info('-'*10+'开始提取向量特征'+'-'*10)
        # get_vector(model_name, corpus_file, output_file, model_type=model_type)
        logger.info('-'*10+'开发入库召回'+'-'*10)
        get_dev_result(model_name, corpus_file, output_file, query_file, recall_result_file, tool_type=tool_type[2],
                       model_type=model_type, batch_size=128)
        logger.info('-'*10+'评测结果'+'-'*10)
        evaluate_result(model_name)
        logger.info('等待几秒'+'*'*20)
        time.sleep(3)



