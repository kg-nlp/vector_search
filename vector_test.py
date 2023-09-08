# -*- coding:utf-8 -*-
'''
# @FileName    :vector_test.py
---------------------------------
# @Time        :2023/9/8 
# @Author      :zhangyj-n
# @Email       :13091375161@163.com
# @zhihu       :https://www.zhihu.com/people/zhangyj-n
# @kg-nlp      :https://kg-nlp.github.io/Algorithm-Project-Manual/
---------------------------------
# 目标任务 :  测试其他代码功能
---------------------------------
'''
from tool.vector_extract import get_single_vector
from model.open_model import get_torch_model,get_paddle_model
from tool.data import convert_example_test,gen_id2corpus,create_dataloader
from tqdm import tqdm
import os
import pandas as pd
import multiprocessing
import numpy as np
from loguru import logger
from functools import partial
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset
import json
from pprint import pprint

current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(current_dir,'data')
'''测试指定模型抽取向量功能'''
def test_extract():
    # model_name = 'Ernie-model/ernie-3.0-medium-zh'  #  768
    # model_name = 'Ernie-model/rocketqa-zh-dureader-query-encoder'  # 768
    model_name = 'Ernie-model/rocketqa-zh-base-query-encoder'  # 768

    corpus_list = ['3.3.3保证项目的检查评定应符合下列规定：1施工方案1）架体搭设应有施工方案，搭设高度超过24m的架体应单独编制安全专项方案，结构设计应进行设计计算，并按规定进行审核、审批；2）搭设高度超过50m的架体，应组织专家对专项方案进行论证，并按专家论证意见组织实施；3）施工方案应完整，能正确指导施工作业。']
    tokenizer, model = get_paddle_model(model_name)
    trans_func = partial(convert_example_test, tokenizer=tokenizer, max_seq_length=256)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # text_segment
    ): [data for data in fn(samples)]
    corpus_list = [{idx: text} for idx, text in enumerate(corpus_list)]
    # print(json.dumps(corpus_list,indent=2,ensure_ascii=False))
    corpus_ds = MapDataset(corpus_list)
    corpus_data_loader = create_dataloader(
        corpus_ds, mode="predict", batch_size=1, batchify_fn=batchify_fn, trans_fn=trans_func
    )
    model.eval()
    with paddle.no_grad():
        for batch_data in corpus_data_loader:
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)
            sequence_output, cls_embedding = model(input_ids, token_type_ids=token_type_ids, position_ids=None,
                                                   attention_mask=None)
            print(input_ids.numpy().tolist()[0])
            print(cls_embedding.numpy().tolist()[0][:20])
            # text_embeddings = F.normalize(cls_embedding, p=2, axis=-1)
            # pprint(cls_embedding.numpy().tolist()[0][:20])
            # total_embedding.append(text_embeddings.numpy())


'''抽取百度预训练微调模型(taskflow)没有测试'''
def baidu_extract():
    from paddlenlp import Taskflow
    import paddle.nn.functional as F
    model_name = 'Ernie-model/rocketqa-zh-dureader-query-encoder'  # 768
    pretrain_model_dir = os.path.abspath(os.path.dirname(__file__)).replace('vector_search',
                                                                            'pretrain_model_file')
    model_name = os.path.join(pretrain_model_dir,model_name)

    # Text feature_extraction with rocketqa-zh-base-query-encoder
    text_encoder = Taskflow("feature_extraction", model=model_name)
    text_embeds = text_encoder(['春天适合种什么花？', '谁有狂三这张高清的?'])
    text_features1 = text_embeds["features"]
    print(text_features1)
    '''
    Tensor(shape=[2, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
        [[ 0.27640465, -0.13405125,  0.00612330, ..., -0.15600294,
            -0.18932408, -0.03029604],
            [-0.12041329, -0.07424965,  0.07895312, ..., -0.17068857,
            0.04485796, -0.18887770]])
    '''
    text_embeds = text_encoder('春天适合种什么菜？')
    text_features2 = text_embeds["features"]
    print(text_features2)
    '''
    Tensor(shape=[1, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
        [[ 0.32578075, -0.02398480, -0.18929179, -0.18639392, -0.04062131,
            0.06708499, -0.04631376, -0.41177100, -0.23074438, -0.23627219,
        ......
    '''
    probs = F.cosine_similarity(text_features1, text_features2)
    print(probs)
    '''
    Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
        [0.86455142, 0.41222256])
    '''

if __name__ == '__main__':

    test_extract()
    # baidu_extract()

