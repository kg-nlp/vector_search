# -*- coding:utf-8 -*-
'''
# @FileName    :open_model.py
---------------------------------
# @Time        :2023/9/6 
# @Author      :zhangyj-n
# @Email       :zhangyj-n@glodon.com
---------------------------------
# 目标任务 :
---------------------------------
'''
import sys
import os
from loguru import logger

torch_model_list = [
'm3e-small',
'm3e-large',
'm3e-base',
'm3e-base-custom',
'bge-small-zh',
'bge-base-zh',
'bge-large-zh',
'text2vec-base-chinese',
'text2vec-large-chinese',
'text2vec-base-chinese-sentence',
'text2vec-base-chinese-paraphrase',
'text2vec-bge-large-chinese',
'Ernie-model/ernie-1.0-base-zh',
'Ernie-model/ernie-1.0-large-zh-cw',
"Ernie-model/ernie-3.0-medium-zh",
'Ernie-model/ernie-3.0-base-zh',
'Ernie-model/rocketqa-zh-dureader-query-encoder',
'Ernie-model/rocketqa-zh-base-query-encoder'
]
paddle_model_list = []

pretrain_model_dir = os.path.abspath(os.path.dirname(__file__)).replace('vector_search','pretrain_model_file')
logger.info('预训练模型目录 '+ pretrain_model_dir)
logger.info('环境目录 '+str(os.listdir(pretrain_model_dir)))

'''加载torch类模型'''
def get_torch_model(model_name):
    from sentence_transformers import SentenceTransformer

    if model_name in torch_model_list:
        return SentenceTransformer(os.path.join(pretrain_model_dir,model_name))
    else:
        raise ValueError('本地模型不存在')

'''加载paddle基础模型'''
def get_paddle_model(model_name):
    import paddle
    from paddlenlp.transformers import AutoModel, AutoTokenizer
    # from .paddle_model import SimCSE
    model_name = os.path.join(pretrain_model_dir,model_name)
    # paddle.set_device('gpu')
    logger.info('loading... '+ model_name)
    # if paddle.distributed.get_world_size() > 1:
    #     paddle.distributed.init_parallel_env()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pretrained_model = AutoModel.from_pretrained(model_name)
    # model = SimCSE(pretrained_model)
    # model = paddle.DataParallel(model)
    # if os.path.isfile(os.path.join(model_name,'model_state.pdparams')):
    #     state_dict = paddle.load(os.path.join(model_name,'model_state.pdparams'))
    #     pretrained_model.set_dict(state_dict)
    # inner_model = model._layers
    # print(inner_model)
    return tokenizer,pretrained_model

'''加载自定义的paddle模型(simcse,in batch negative训练过)'''
def get_paddle_simcse_model(model_name):
    import paddle
    from paddlenlp.transformers import AutoModel, AutoTokenizer
    from model.paddle_model import SimCSE
    model_name = os.path.join(pretrain_model_dir,model_name)
    logger.info('loading... '+ model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pretrained_model = AutoModel.from_pretrained(model_name)
    model = SimCSE(pretrained_model)
    state_dict = paddle.load(os.path.join(model_name, 'model_state.pdparams'))
    model.set_dict(state_dict)
    return tokenizer, model

if __name__ == '__main__':
    # import torch
    # print(torch.cuda.is_available())
    # get_torch_model('bge-large-zh')
    # get_paddle_model('')
    pass